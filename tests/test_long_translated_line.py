"""One runaway line must not discard a finished translation.

SubtitleLine caps a line at 2000 characters. That cap is applied to the
response as well as the request, so a model that loops and emits one cue past
the limit fails response validation, and a job that translated the whole film
is reported failed and delivers nothing.

Seen for real: tencent/hy-mt2-1.8b translated all 1338 cues of a feature film,
17 of 17 batches, and the result was thrown away with
"String should have at most 2000 characters".
"""

import json

import httpx
import pytest

from subtitle_translator.api.models import SubtitleLine
from subtitle_translator.config import Settings
from subtitle_translator.core.batch_sizing import get_batch_size_resolver
from subtitle_translator.core.translator import SubtitleTranslator, map_translations_to_lines
from subtitle_translator.providers.openrouter import OpenRouterProvider
from subtitle_translator.queue.job_manager import JobManager, JobStatus, JobType
from subtitle_translator.queue.worker import process_content_translation_job

LIMIT = 2000


@pytest.fixture(autouse=True)
def reset_sizing():
    get_batch_size_resolver().reset()
    yield
    get_batch_size_resolver().reset()


def settings(**kwargs):
    return Settings(
        _env_file=None,
        openrouter_api_key="synthetic-test-only",
        max_retries=1,
        retry_delay=0,
        parallel_batches_per_job=1,
        **kwargs,
    )


class TestMappingARunawayLine:
    def test_a_line_past_the_limit_does_not_raise(self):
        originals = [
            SubtitleLine(position=0, line="Wake up, Neo."),
            SubtitleLine(position=1, line="Follow the white rabbit."),
        ]
        translations = [
            {"index": "0", "content": "Ébredj, Neo."},
            {"index": "1", "content": "hurok " * 1000},
        ]

        mapped = map_translations_to_lines(originals, translations, "hu", settings())

        assert [line.position for line in mapped] == [0, 1]
        assert mapped[0].line == "Ébredj, Neo."

    def test_the_runaway_position_keeps_its_source_text(self):
        originals = [SubtitleLine(position=0, line="Follow the white rabbit.")]
        translations = [{"index": "0", "content": "x" * (LIMIT + 1)}]

        mapped = map_translations_to_lines(originals, translations, "hu", settings())

        assert mapped[0].line == "Follow the white rabbit."

    def test_a_line_exactly_at_the_limit_is_kept(self):
        originals = [SubtitleLine(position=0, line="Follow the white rabbit.")]
        translations = [{"index": "0", "content": "y" * LIMIT}]

        mapped = map_translations_to_lines(originals, translations, "hu", settings())

        assert mapped[0].line == "y" * LIMIT

    def test_it_says_which_position_was_dropped(self, caplog):
        originals = [SubtitleLine(position=41, line="Follow the white rabbit.")]
        translations = [{"index": "41", "content": "z" * (LIMIT + 1)}]

        with caplog.at_level("WARNING"):
            map_translations_to_lines(originals, translations, "hu", settings())

        assert any("41" in record.message for record in caplog.records), caplog.text


@pytest.mark.asyncio
async def test_a_job_with_one_runaway_line_still_delivers_the_file(monkeypatch):
    """The whole point: 2 good cues out of 3 beats nothing at all."""

    async def send(request):
        lines = json.loads(json.loads(request.content)["messages"][-1]["content"])
        translations = []
        for line in lines:
            content = "loop " * 600 if line["index"] == "1" else "leforditva"
            translations.append({"index": line["index"], "content": content})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps({"translations": translations})}}],
                "usage": {"total_tokens": 10, "cost": 0.01},
            },
        )

    provider = OpenRouterProvider(settings())
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(send), base_url="https://fake"
    )
    provider._model_params_fetched = True
    translator = SubtitleTranslator(provider=provider, settings=provider.settings)
    manager = JobManager()
    job_id = await manager.submit_job(
        job_type=JobType.TRANSLATE_CONTENT,
        request_data={
            "sourceLanguage": "English",
            "targetLanguage": "Hungarian",
            "lines": [
                {"position": 0, "line": "Wake up, Neo."},
                {"position": 1, "line": "Follow the white rabbit."},
                {"position": 2, "line": "Knock, knock."},
            ],
        },
    )
    manager.set_job_processing(job_id)
    try:
        await process_content_translation_job(manager, job_id, translator)
    finally:
        await provider.close()

    job = manager.get_job(job_id)
    assert job.status is not JobStatus.FAILED, job.error
    delivered = {line["position"]: line["line"] for line in job.result["lines"]}
    assert delivered[0] == "leforditva"
    assert delivered[2] == "leforditva"
    assert delivered[1] == "Follow the white rabbit."
