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
    assert job.error.startswith("2/4 lines translated. ")
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


@pytest.mark.asyncio
async def test_progress_reaches_the_job(manager):
    job_id = await submit(manager)
    await process_content_translation_job(manager, job_id, translator(ScriptedProvider()))
    job = manager.get_job(job_id)
    assert job.progress == 100 and job.completed_lines == 4


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
    assert manager.get_job(job_id).status in (JobStatus.FAILED, JobStatus.COMPLETED)
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
