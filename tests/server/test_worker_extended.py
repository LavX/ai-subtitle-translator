"""Extended tests for worker.py: the outer error handler and config extraction."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from subtitle_translator.server.queue.job_manager import JobManager, JobStatus, JobType
from subtitle_translator.server.queue.worker import (
    _extract_config_override_from_dict,
    process_file_translation_job,
)


@pytest.fixture
def manager():
    return JobManager(max_concurrent=2, max_jobs=10)


@pytest.fixture
def mock_translator():
    translator = AsyncMock()
    translator.settings = MagicMock()
    translator.settings.openrouter_default_model = "test-model"
    translator.settings.batch_size = 25
    translator.settings.parallel_batches_per_job = 4
    translator.settings.max_retries = 2
    translator.settings.retry_delay = 0.01
    translator.settings.is_rtl_language = MagicMock(return_value=False)
    translator.provider = AsyncMock()
    return translator


class TestFileJobExceptionHandler:
    """Tests for the outer exception handler in file translation."""

    @pytest.mark.asyncio
    async def test_unexpected_exception_sets_failed(self, manager, mock_translator):
        """An unexpected exception after the parse is caught by the outer handler."""
        parser = MagicMock()
        parser.parse.return_value = [MagicMock()]
        parser.extract_lines_for_translation.side_effect = ValueError("unexpected format")

        job_id = await manager.submit_job(
            request_data={
                "content": "valid srt content",
                "sourceLanguage": "en",
                "targetLanguage": "es",
            },
            job_type=JobType.TRANSLATE_FILE,
        )
        manager.set_job_processing(job_id)
        with patch("subtitle_translator.server.queue.worker.SRTParser", return_value=parser):
            await process_file_translation_job(manager, job_id, mock_translator)

        job = manager.get_job(job_id)
        assert job.status == JobStatus.FAILED
        assert "unexpected format" in job.error


class TestExtractConfigParallelBatches:
    """Test that parallelBatches is recognized by _extract_config_override_from_dict."""

    def test_parallel_batches_alias_is_recognized(self):
        """A parallelism-only file config must survive extraction."""
        result = _extract_config_override_from_dict({"parallelBatches": 3})
        assert result.parallel_batches == 3

    def test_parallel_batches_with_recognized_key(self):
        """parallelBatches works when combined with a recognized key."""
        result = _extract_config_override_from_dict({"model": "gpt-4", "parallelBatches": 3})
        assert result is not None
        assert result.parallel_batches == 3
        assert result.model == "gpt-4"
