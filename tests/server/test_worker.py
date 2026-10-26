"""Tests for worker functions (best effort coverage)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from subtitle_translator.core.models import TranslationConfig
from subtitle_translator.server.queue.job_manager import JobManager, JobStatus, JobType
from subtitle_translator.server.queue.worker import (
    _extract_config_override,
    _extract_config_override_from_dict,
    job_worker_handler,
    process_content_translation_job,
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


class TestExtractConfigOverride:
    """Tests for config override extraction helpers."""

    def test_extract_with_parsed_config(self):
        config = TranslationConfig(model="test")
        result = _extract_config_override(config, {})
        assert result is config

    def test_extract_from_raw_dict(self):
        result = _extract_config_override(None, {"config": {"model": "test"}})
        assert result is not None
        assert result.model == "test"

    def test_extract_none_when_no_config(self):
        assert _extract_config_override(None, {}) is None

    def test_extract_from_dict_none(self):
        assert _extract_config_override_from_dict(None) is None

    def test_extract_from_dict_not_dict(self):
        assert _extract_config_override_from_dict("not a dict") is None

    def test_extract_from_dict_empty(self):
        assert _extract_config_override_from_dict({}) is None

    def test_extract_from_dict_valid(self):
        result = _extract_config_override_from_dict({"model": "test-model", "temperature": 0.5})
        assert result is not None
        assert result.model == "test-model"

    def test_extract_from_dict_invalid_fields(self):
        result = _extract_config_override_from_dict({"invalid_field_xyz": True})
        assert result is None

    def test_extract_from_dict_validation_error(self):
        result = _extract_config_override_from_dict({"temperature": "not-a-number"})
        assert result is None


class TestProcessContentTranslationJob:
    """Tests for content translation job processing."""

    @pytest.mark.asyncio
    async def test_missing_job(self, manager, mock_translator):
        await process_content_translation_job(manager, "nonexistent", mock_translator)
        # Should not raise

    @pytest.mark.asyncio
    async def test_empty_lines(self, manager, mock_translator):
        job_id = await manager.submit_job(
            request_data={
                "sourceLanguage": "en",
                "targetLanguage": "hu",
                "lines": [],
            },
            job_type=JobType.TRANSLATE_CONTENT,
        )
        manager.set_job_processing(job_id)
        await process_content_translation_job(manager, job_id, mock_translator)
        job = manager.get_job(job_id)
        assert job.status == JobStatus.COMPLETED

class TestProcessFileTranslationJob:
    """Tests for file translation job processing."""

    @pytest.mark.asyncio
    async def test_missing_job(self, manager, mock_translator):
        await process_file_translation_job(manager, "nonexistent", mock_translator)

    @pytest.mark.asyncio
    async def test_empty_content(self, manager, mock_translator):
        job_id = await manager.submit_job(
            request_data={
                "content": "",
                "sourceLanguage": "en",
                "targetLanguage": "hu",
            },
            job_type=JobType.TRANSLATE_FILE,
        )
        manager.set_job_processing(job_id)
        await process_file_translation_job(manager, job_id, mock_translator)
        job = manager.get_job(job_id)
        assert job.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_invalid_srt_content(self, manager, mock_translator):
        parser = MagicMock()
        parser.parse.side_effect = Exception("parse error")

        job_id = await manager.submit_job(
            request_data={
                "content": "not srt",
                "sourceLanguage": "en",
                "targetLanguage": "hu",
            },
            job_type=JobType.TRANSLATE_FILE,
        )
        manager.set_job_processing(job_id)
        with patch("subtitle_translator.server.queue.worker.SRTParser", return_value=parser):
            await process_file_translation_job(manager, job_id, mock_translator)
        job = manager.get_job(job_id)
        assert job.status == JobStatus.FAILED
        assert "Invalid SRT" in job.error

    @pytest.mark.asyncio
    async def test_empty_entries(self, manager, mock_translator):
        parser = MagicMock()
        parser.parse.return_value = []

        job_id = await manager.submit_job(
            request_data={
                "content": "1\n00:00:01,000 --> 00:00:02,000\nHi\n\n",
                "sourceLanguage": "en",
                "targetLanguage": "hu",
            },
            job_type=JobType.TRANSLATE_FILE,
        )
        manager.set_job_processing(job_id)
        with patch("subtitle_translator.server.queue.worker.SRTParser", return_value=parser):
            await process_file_translation_job(manager, job_id, mock_translator)
        job = manager.get_job(job_id)
        assert job.status == JobStatus.COMPLETED
        assert job.result["subtitle_count"] == 0

class TestJobWorkerHandler:
    """Tests for the main worker handler routing."""

    @pytest.mark.asyncio
    async def test_routes_to_content(self, manager):
        job_id = await manager.submit_job(
            request_data={
                "sourceLanguage": "en",
                "targetLanguage": "hu",
                "lines": [],
            },
            job_type=JobType.TRANSLATE_CONTENT,
        )
        manager.set_job_processing(job_id)

        with patch("subtitle_translator.server.queue.worker.get_translator") as mock_get:
            translator = AsyncMock()
            mock_get.return_value = translator
            with patch(
                "subtitle_translator.server.queue.worker.process_content_translation_job"
            ) as mock_proc:
                mock_proc.return_value = None
                await job_worker_handler(manager, job_id, JobType.TRANSLATE_CONTENT)
                mock_proc.assert_called_once_with(manager, job_id, translator)

    @pytest.mark.asyncio
    async def test_routes_to_file(self, manager):
        job_id = await manager.submit_job(
            request_data={
                "content": "srt",
                "sourceLanguage": "en",
                "targetLanguage": "hu",
            },
            job_type=JobType.TRANSLATE_FILE,
        )
        manager.set_job_processing(job_id)

        with patch("subtitle_translator.server.queue.worker.get_translator") as mock_get:
            translator = AsyncMock()
            mock_get.return_value = translator
            with patch(
                "subtitle_translator.server.queue.worker.process_file_translation_job"
            ) as mock_proc:
                mock_proc.return_value = None
                await job_worker_handler(manager, job_id, JobType.TRANSLATE_FILE)
                mock_proc.assert_called_once_with(manager, job_id, translator)

    @pytest.mark.asyncio
    async def test_unknown_job_type(self, manager):
        job_id = await manager.submit_job(
            request_data={},
            job_type=JobType.TRANSLATE_CONTENT,
        )
        manager.set_job_processing(job_id)

        with patch("subtitle_translator.server.queue.worker.get_translator") as mock_get:
            mock_get.return_value = AsyncMock()
            # Pass an invalid type string to trigger else branch
            await job_worker_handler(manager, job_id, "unknown_type")
            job = manager.get_job(job_id)
            assert job.status == JobStatus.FAILED


@pytest.mark.parametrize("field", ["requestTimeout", "request_timeout"])
def test_timeout_only_config_survives_worker_extraction(field):
    config = _extract_config_override_from_dict({field: 600})
    assert config is not None
    assert config.request_timeout == 600
