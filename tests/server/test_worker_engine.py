"""Queued jobs through the engine: the same states, payloads and messages as 2.1.1."""

import pytest

from subtitle_translator.core.errors import InvalidResponseError, QuotaExhaustedError
from subtitle_translator.core.settings import CoreSettings
from subtitle_translator.server.queue.job_manager import JobManager, JobStatus, JobType
from subtitle_translator.server.queue.worker import (
    process_content_translation_job,
    process_file_translation_job,
)
from subtitle_translator.server.runtime import EngineTranslator, single_provider_runtime
from tests.core.engine_fakes import ScriptedProvider

SETTINGS = CoreSettings(openrouter_api_key="server", max_retries=1, retry_delay=0.0, batch_size=2,
                        parallel_batches_per_job=1, request_timeout=5.0)
SRT = "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n2\n00:00:03,000 --> 00:00:04,000\nBye\n\n"


def translator(provider):
    return EngineTranslator(single_provider_runtime(provider, SETTINGS))


async def submit(manager, lines=None, job_type=JobType.TRANSLATE_CONTENT, key=None, **extra):
    data = {"sourceLanguage": "en", "targetLanguage": "hu", **extra}
    if job_type is JobType.TRANSLATE_CONTENT:
        data["lines"] = lines or [{"position": i, "line": f"l{i}"} for i in range(4)]
    job_id = await manager.submit_job(request_data=data, job_type=job_type, api_key_override=key)
    manager.set_job_processing(job_id)
    return job_id


@pytest.fixture
def manager():
    return JobManager(max_concurrent=2, max_jobs=10)


@pytest.mark.asyncio
async def test_a_clean_content_job_completes_with_tokens_and_model(manager):
    job_id = await submit(manager)
    await process_content_translation_job(manager, job_id, translator(ScriptedProvider()))
    job = manager.get_job(job_id)
    assert job.status == JobStatus.COMPLETED
    assert [item["line"] for item in job.result["lines"]] == ["Tl0", "Tl1", "Tl2", "Tl3"]
    assert job.result["tokens_used"] == 20 and job.result["model_used"]


@pytest.mark.asyncio
async def test_a_content_failure_is_partial_with_the_2_1_1_message(manager):
    bad = InvalidResponseError("garbled")
    job_id = await submit(manager)
    await process_content_translation_job(manager, job_id, translator(ScriptedProvider(["ok", bad, bad])))
    job = manager.get_job(job_id)
    assert job.status == JobStatus.PARTIAL
    assert job.error == "2/4 lines translated. 1 of 2 batches failed: garbled"
    assert [item["line"] for item in job.result["lines"]] == ["Tl0", "Tl1", "l2", "l3"]


@pytest.mark.asyncio
async def test_repeated_positions_count_as_lines(manager):
    bad = InvalidResponseError("garbled")
    lines = [{"position": 0, "line": "a"}, {"position": 0, "line": "a"}, {"position": 1, "line": "b"}, {"position": 2, "line": "c"}]
    job_id = await submit(manager, lines)
    await process_content_translation_job(manager, job_id, translator(ScriptedProvider(["ok", bad, bad])))
    # both lines at position 0 came back: they count as two lines, not one position
    assert manager.get_job(job_id).error.startswith("2/4 lines translated. ")


@pytest.mark.asyncio
async def test_quota_after_some_work_is_partial_and_nothing_first_is_failed(manager):
    job_id = await submit(manager)
    await process_content_translation_job(manager, job_id, translator(ScriptedProvider(["ok", QuotaExhaustedError("spent")])))
    assert manager.get_job(job_id).status == JobStatus.PARTIAL
    job_id = await submit(manager)
    await process_content_translation_job(manager, job_id, translator(ScriptedProvider([QuotaExhaustedError("spent")])))
    job = manager.get_job(job_id)
    assert job.status == JobStatus.FAILED and "spent" in job.error


@pytest.mark.asyncio
async def test_a_restored_key_pins_the_job_to_it(manager):
    provider = ScriptedProvider()
    job_id = await submit(manager, key="sk-user")
    await process_content_translation_job(manager, job_id, translator(provider))
    assert manager.get_job(job_id).status == JobStatus.COMPLETED
    assert provider.batches                                   # served by the pinned request connection


def record_progress(manager):
    """Every progress update the job receives: (percent, message, lines, batches, total batches)."""
    updates = []
    original = manager.update_progress

    def update(job_id, progress, message="", **kwargs):
        updates.append(
            (
                progress,
                message,
                kwargs.get("completed_lines"),
                kwargs.get("completed_batches"),
                kwargs.get("total_batches"),
            )
        )
        original(job_id, progress, message, **kwargs)

    manager.update_progress = update
    return updates


@pytest.mark.asyncio
async def test_progress_reaches_the_job(manager):
    updates = record_progress(manager)
    job_id = await submit(manager)
    await process_content_translation_job(manager, job_id, translator(ScriptedProvider()))
    job = manager.get_job(job_id)
    assert job.progress == 100 and job.completed_lines == 4
    settled = [u for u in updates if u[1].startswith("Translated ")]
    assert [(u[0], u[1], u[2], u[3], u[4]) for u in settled] == [
        (0, "Translated 0/4 lines (0/2 batches)", 0, 0, 2),
        (50, "Translated 2/4 lines (1/2 batches)", 2, 1, 2),
        (100, "Translated 4/4 lines (2/2 batches)", 4, 2, 2),
    ]
    assert any(u[1].startswith("Batch 2: request in progress") and u[0] == 50 for u in updates)
    percents = [u[0] for u in updates]
    assert percents == sorted(percents)


@pytest.mark.asyncio
async def test_a_failed_batch_shows_in_the_progress_messages(manager):
    bad = InvalidResponseError("garbled")
    updates = record_progress(manager)
    job_id = await submit(manager)
    await process_content_translation_job(manager, job_id, translator(ScriptedProvider(["ok", bad, bad])))
    assert manager.get_job(job_id).status == JobStatus.PARTIAL
    messages = [u[1] for u in updates]
    assert "Translated 2/4 lines (1/2 batches)" in messages
    assert any(m.startswith("Batch 2: retry 1 after invalid response") for m in messages)
    final = updates[-1]
    assert final[1] == "Translated 2/4 lines (2/2 batches, 1 failed)"
    assert final[0] == 100 and final[2] == 2 and final[3] == 2


@pytest.mark.asyncio
async def test_a_file_job_translates_and_composes(manager):
    job_id = await submit(manager, job_type=JobType.TRANSLATE_FILE, content=SRT)
    await process_file_translation_job(manager, job_id, translator(ScriptedProvider()))
    job = manager.get_job(job_id)
    assert job.status == JobStatus.COMPLETED and "THello" in job.result["content"]
    assert job.result["subtitle_count"] == 2


@pytest.mark.asyncio
async def test_a_file_job_into_hebrew_gets_rtl_marks_once(manager):
    job_id = await submit(manager, job_type=JobType.TRANSLATE_FILE, content=SRT, targetLanguage="he")
    await process_file_translation_job(manager, job_id, translator(ScriptedProvider()))
    content = manager.get_job(job_id).result["content"]
    assert content.count("‫") == 2


@pytest.mark.asyncio
async def test_invalid_srt_and_empty_content_fail_like_2_1_1(manager):
    job_id = await submit(manager, job_type=JobType.TRANSLATE_FILE, content="not srt at all")
    await process_file_translation_job(manager, job_id, translator(ScriptedProvider()))
    job = manager.get_job(job_id)
    assert job.status == JobStatus.FAILED
    assert job.error.startswith("Invalid SRT content: Failed to parse SRT content")
    job_id = await submit(manager, job_type=JobType.TRANSLATE_FILE, content="   ")
    await process_file_translation_job(manager, job_id, translator(ScriptedProvider()))
    assert manager.get_job(job_id).error == "SRT content is required"


@pytest.mark.asyncio
async def test_the_sync_translator_surface():
    from subtitle_translator.core.models import SubtitleLine, TranslateContentRequest

    t = translator(ScriptedProvider())
    content = await t.translate_content(
        TranslateContentRequest(sourceLanguage="en", targetLanguage="hu", lines=[SubtitleLine(position=1, line="Hi")])
    )
    assert content.success and content.lines[0].line == "THi"
    file_result = await t.translate_file(SRT, "en", "hu")
    assert file_result.success and file_result.subtitle_count == 2


@pytest.mark.asyncio
async def test_request_config_is_logged_without_key_material(manager, caplog):
    import logging

    caplog.set_level(logging.DEBUG)
    job_id = await submit(manager, config={"apiKey": "sk-visible-nowhere", "model": "x/y"})
    await process_content_translation_job(manager, job_id, translator(ScriptedProvider()))
    assert "sk-visible-nowhere" not in caplog.text and "***" in caplog.text


@pytest.mark.asyncio
async def test_every_batch_failing_on_content_fails_the_job(manager):
    bad = InvalidResponseError("garbled")
    job_id = await submit(manager)
    await process_content_translation_job(manager, job_id, translator(ScriptedProvider([bad] * 10)))
    job = manager.get_job(job_id)
    assert job.status == JobStatus.FAILED
    assert job.error.startswith("All 2 batches failed") and "garbled" in job.error
    assert not job.result


@pytest.mark.asyncio
async def test_a_file_where_every_batch_fails_on_content_fails_the_job(manager):
    bad = InvalidResponseError("garbled")
    job_id = await submit(manager, job_type=JobType.TRANSLATE_FILE, content=SRT)
    await process_file_translation_job(manager, job_id, translator(ScriptedProvider([bad] * 10)))
    job = manager.get_job(job_id)
    assert job.status == JobStatus.FAILED and "garbled" in job.error


@pytest.mark.asyncio
async def test_the_sync_surface_returns_no_lines_when_every_batch_fails():
    from subtitle_translator.core.models import SubtitleLine, TranslateContentRequest

    bad = InvalidResponseError("garbled")
    t = translator(ScriptedProvider([bad] * 10))
    content = await t.translate_content(
        TranslateContentRequest(sourceLanguage="en", targetLanguage="hu", lines=[SubtitleLine(position=1, line="Hi")])
    )
    assert content.success is False and content.lines == [] and "garbled" in content.error
    file_result = await translator(ScriptedProvider([bad] * 10)).translate_file(SRT, "en", "hu")
    assert file_result.success is False and file_result.content == ""


@pytest.mark.asyncio
async def test_a_file_job_config_with_only_a_backend_pin_is_kept(manager):
    from subtitle_translator.server.queue.worker import _extract_config_override_from_dict

    assert _extract_config_override_from_dict({"backend": "gemini"}).backend == "gemini"


@pytest.mark.asyncio
async def test_a_translation_the_rtl_marks_push_past_the_line_cap_does_not_fail_the_job(manager):
    from subtitle_translator.core.models import MAX_LINE_CHARS
    from subtitle_translator.core.provider import TranslationResult

    def long_first(batch):
        return TranslationResult(
            translations=[
                {"index": line["index"], "content": "x" * (MAX_LINE_CHARS - 1) if line["index"] == "0" else "ok"}
                for line in batch.lines
            ],
            model_used="scripted-model",
            total_tokens=10,
            cost=0.01,
        )

    job_id = await submit(manager, targetLanguage="ar")
    await process_content_translation_job(manager, job_id, translator(ScriptedProvider([long_first])))
    job = manager.get_job(job_id)
    assert job.status == JobStatus.PARTIAL, job.error
    assert job.result["lines"][0]["line"] == "l0"      # the source text, as in 2.1.1
    assert job.error.startswith("3/4 lines translated. ")


def _provider_returning(text):
    from subtitle_translator.core.provider import TranslationResult

    def reply(batch):
        return TranslationResult(
            translations=[{"index": line["index"], "content": text} for line in batch.lines],
            model_used="scripted-model",
            total_tokens=10,
            cost=0.01,
        )

    return reply


LONG = "this translated line is much longer than a single subtitle line should be"


@pytest.mark.asyncio
async def test_a_completed_file_job_splits_long_subtitles(manager):
    job_id = await submit(manager, job_type=JobType.TRANSLATE_FILE, content=SRT)
    await process_file_translation_job(manager, job_id, translator(ScriptedProvider([_provider_returning(LONG)] * 2)))
    job = manager.get_job(job_id)
    assert job.status == JobStatus.COMPLETED
    blocks = [b for b in job.result["content"].strip().split("\n\n") if b]
    assert all(len(b.split("\n")) == 4 for b in blocks)           # number, time, two text lines
    assert "this translated line is much longer than a\nsingle subtitle line should be" in job.result["content"]
    assert all(len(text) <= 42 for b in blocks for text in b.split("\n")[2:])


@pytest.mark.asyncio
async def test_a_partial_file_job_leaves_long_subtitles_unsplit(manager):
    bad = InvalidResponseError("garbled")
    three = SRT + "3\n00:00:05,000 --> 00:00:06,000\nThird\n\n"
    job_id = await submit(manager, job_type=JobType.TRANSLATE_FILE, content=three)
    await process_file_translation_job(
        manager, job_id, translator(ScriptedProvider([_provider_returning(LONG), bad, bad]))
    )
    job = manager.get_job(job_id)
    assert job.status == JobStatus.PARTIAL
    assert "\n" + LONG + "\n" in job.result["content"]


def _watch_translate(runtime):
    seen = {}
    real = runtime.engine.translate

    async def translate(job, connections, *, pinned=None, events=None):
        seen["connections"], seen["pinned"] = connections, pinned
        return await real(job, connections, pinned=pinned, events=events)

    runtime.engine.translate = translate
    return seen


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["content", "file"])
async def test_a_job_key_is_restored_into_a_config_that_has_none(manager, kind):
    runtime = single_provider_runtime(ScriptedProvider(), SETTINGS)
    seen = _watch_translate(runtime)
    job_type = JobType.TRANSLATE_CONTENT if kind == "content" else JobType.TRANSLATE_FILE
    extra = {"content": SRT} if kind == "file" else {}
    job_id = await submit(manager, job_type=job_type, key="sk-job-key", config={"model": "gpt-4"}, **extra)
    work = process_content_translation_job if kind == "content" else process_file_translation_job
    await work(manager, job_id, EngineTranslator(runtime))
    assert manager.get_job(job_id).status == JobStatus.COMPLETED
    assert seen["pinned"] is not None
    pinned = next(c for c in seen["connections"] if c.id == seen["pinned"])
    assert pinned.secrets == {"api_key": "sk-job-key"} and pinned.settings["model"] == "gpt-4"


@pytest.mark.asyncio
async def test_a_file_partial_counts_repeated_cue_numbers_as_lines(manager):
    bad = InvalidResponseError("garbled")
    srt = (
        "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n"
        "1\n00:00:03,000 --> 00:00:04,000\nHello again\n\n"
        "2\n00:00:05,000 --> 00:00:06,000\nWorld\n\n"
    )
    job_id = await submit(manager, job_type=JobType.TRANSLATE_FILE, content=srt)
    await process_file_translation_job(manager, job_id, translator(ScriptedProvider(["ok", bad, bad])))
    job = manager.get_job(job_id)
    assert job.status == JobStatus.PARTIAL
    assert job.error.startswith("2/3 lines translated. ")


@pytest.mark.asyncio
async def test_reasoning_and_provider_settings_are_logged_without_keys(manager, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="subtitle_translator.server.queue.worker")
    config = {
        "apiKey": "SECRET-request-key",
        "reasoning": {"effort": "medium"},
        "provider": {"order": ["deepinfra"], "allowFallbacks": False},
    }
    job_id = await submit(manager, config=config)
    await process_content_translation_job(manager, job_id, translator(ScriptedProvider()))
    assert manager.get_job(job_id).status == JobStatus.COMPLETED
    messages = [r.getMessage() for r in caplog.records]
    assert sum("Reasoning config" in m for m in messages) == 1
    assert sum("Provider config configured" in m for m in messages) == 1
    assert "SECRET" not in " ".join(messages)


UNNUMBERED_SRT = (
    "00:00:01,000 --> 00:00:02,000\nHello\n\n"
    "00:00:03,000 --> 00:00:04,000\nBye\n\n"
    "00:00:05,000 --> 00:00:06,000\nAgain\n\n"
)


@pytest.mark.asyncio
async def test_a_file_job_translates_cues_that_carry_no_number(manager):
    provider = ScriptedProvider()
    job_id = await submit(manager, job_type=JobType.TRANSLATE_FILE, content=UNNUMBERED_SRT)
    await process_file_translation_job(manager, job_id, translator(provider))
    job = manager.get_job(job_id)
    assert job.status == JobStatus.COMPLETED, job.error
    content = job.result["content"]
    assert all(text in content for text in ("THello", "TBye", "TAgain"))
    assert job.result["subtitle_count"] == 3


@pytest.mark.asyncio
async def test_the_sync_translator_translates_cues_that_carry_no_number():
    result = await translator(ScriptedProvider()).translate_file(UNNUMBERED_SRT, "en", "hu")
    assert result.success and result.subtitle_count == 3
    assert all(text in result.content for text in ("THello", "TBye", "TAgain"))


@pytest.mark.asyncio
async def test_repeated_cue_numbers_do_not_merge_translations(manager):
    repeated = SRT.replace("2\n00:00:03", "1\n00:00:03")
    result = await translator(ScriptedProvider()).translate_file(repeated, "en", "hu")
    assert result.success and "THello" in result.content and "TBye" in result.content


def _failover_translator(a, b):
    from subtitle_translator.core.backends import Connection
    from subtitle_translator.core.engine import TranslationEngine
    from subtitle_translator.server.runtime import Runtime
    from tests.core.engine_fakes import ScriptedBackend, registry_with

    registry = registry_with(ScriptedBackend({"a": a, "b": b}, batch_size=2))
    connections = [Connection("a", "scripted"), Connection("b", "scripted")]
    engine = TranslationEngine(registry, SETTINGS)
    return EngineTranslator(Runtime(SETTINGS, SETTINGS, registry, engine, connections, b))


@pytest.mark.asyncio
async def test_job_totals_after_a_failover_include_every_attempt(manager):
    a = ScriptedProvider(["ok", QuotaExhaustedError("spent", retry_after=600)], tokens=10, cost=0.01)
    b = ScriptedProvider(tokens=20, cost=0.02)
    job_id = await submit(manager)
    await process_content_translation_job(manager, job_id, _failover_translator(a, b))
    job = manager.get_job(job_id)
    assert job.status == JobStatus.COMPLETED
    assert job.tokens_used == 10 + 20 == job.result["tokens_used"]
    assert job.total_cost == pytest.approx(0.03)


@pytest.mark.parametrize("restriction", [{"only": ["safe"]}, {"ignore": ["bad"]}, {"dataCollection": "deny"}])
def test_an_unreadable_config_with_provider_restrictions_is_refused_not_dropped(restriction):
    from subtitle_translator.server.queue.worker import _extract_config_override_from_dict

    with pytest.raises(ValueError):
        _extract_config_override_from_dict(
            {"model": "x/y", "temperature": "hot", "provider": restriction}
        )
    # Without a restriction the unreadable config is still ignored, as before.
    assert _extract_config_override_from_dict({"model": "x/y", "temperature": "hot"}) is None


@pytest.mark.asyncio
async def test_a_failover_counts_repeated_positions_as_lines_in_the_final_progress(manager):
    lines = [{"position": 0, "line": "a"}, {"position": 0, "line": "a"}, {"position": 1, "line": "b"}, {"position": 2, "line": "c"}]
    a = ScriptedProvider(["ok", QuotaExhaustedError("spent", retry_after=600)])
    b = ScriptedProvider()
    updates = record_progress(manager)
    job_id = await submit(manager, lines)
    await process_content_translation_job(manager, job_id, _failover_translator(a, b))
    job = manager.get_job(job_id)
    assert job.status == JobStatus.COMPLETED
    # Position 0 was translated before the failover and stands for two lines.
    assert job.completed_lines == 4 and updates[-1][2] == 4
    assert all(u[2] is None or u[2] <= 4 for u in updates)
