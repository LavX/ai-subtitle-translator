"""Extended tests for worker.py to cover missing lines."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from subtitle_translator.api.models import SubtitleLine
from subtitle_translator.core.batch_processor import (
    BatchProgress,
)
from subtitle_translator.queue.job_manager import JobManager, JobStatus, JobType
from subtitle_translator.queue.worker import (
    _extract_config_override_from_dict,
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
    translator._srt_parser = MagicMock()
    return translator


def _successful_result(translations=None, tokens=100, model="test-model"):
    """Helper to build a successful BatchProcessingResult mock."""
    result = MagicMock()
    result.success = True
    result.all_translations = translations or [{"index": "0", "content": "Hola"}]
    result.total_tokens = tokens
    result.model_used = model
    result.batch_results = [MagicMock(success=True)]
    return result


def _partial_result(translations=None, tokens=80, model="test-model"):
    """Helper to build a partial-failure BatchProcessingResult mock."""
    result = MagicMock()
    result.success = False
    result.all_translations = translations or [{"index": "0", "content": "Hola"}]
    result.total_tokens = tokens
    result.model_used = model
    result.batch_results = [
        MagicMock(success=True, error=None),
        MagicMock(success=False, error="Batch 2 rate limited"),
    ]
    return result


def _all_failed_result():
    """Helper to build an all-failed BatchProcessingResult mock."""
    result = MagicMock()
    result.success = False
    result.all_translations = []
    result.total_tokens = 0
    result.model_used = "test-model"
    result.batch_results = [
        MagicMock(success=False, error="API error"),
    ]
    return result


# ---------------------------------------------------------------------------
# Content job: config override logging (lines 56-62)
# ---------------------------------------------------------------------------


class TestContentJobConfigLogging:
    """Tests that config with sensitive keys gets masked in logs."""

    @pytest.mark.asyncio
    async def test_config_with_sensitive_keys_masked(self, manager, mock_translator):
        """Lines 56-62: safe_config masks keys containing 'key', 'secret', 'password'."""
        mock_result = _successful_result()

        with patch("subtitle_translator.queue.worker.BatchProcessor") as MockBP:
            processor = AsyncMock()
            processor.process_all_batches = AsyncMock(return_value=mock_result)
            MockBP.return_value = processor

            with patch("subtitle_translator.queue.worker.map_translations_to_lines") as mock_map:
                mock_map.return_value = [SubtitleLine(position=1, line="Hola")]

                job_id = await manager.submit_job(
                    request_data={
                        "sourceLanguage": "en",
                        "targetLanguage": "es",
                        "lines": [{"position": 1, "line": "Hello"}],
                        "config": {
                            "apiKey": "sk-secret-123",
                            "model": "gpt-4",
                            "secret_token": "abc",
                            "password": "",
                        },
                    },
                    job_type=JobType.TRANSLATE_CONTENT,
                )
                manager.set_job_processing(job_id)

                with patch("subtitle_translator.queue.worker.logger") as mock_logger:
                    await process_content_translation_job(manager, job_id, mock_translator)

                    # Verify info was called with the safe config (keys masked)
                    info_calls = [str(c) for c in mock_logger.info.call_args_list]
                    config_log = [c for c in info_calls if "Request config" in c]
                    assert len(config_log) == 1
                    assert "***" in config_log[0]
                    assert "sk-secret-123" not in config_log[0]
                    # model should appear unmasked
                    assert "gpt-4" in config_log[0]

                job = manager.get_job(job_id)
                assert job.status == JobStatus.COMPLETED


# ---------------------------------------------------------------------------
# Content job: reasoning and provider config logging (lines 71-72, 82, 87)
# ---------------------------------------------------------------------------


class TestContentJobReasoningAndProviderLogging:
    """Tests for reasoning/provider config logging branches."""

    @pytest.mark.asyncio
    async def test_reasoning_config_logged(self, manager, mock_translator):
        """Lines 82-85: reasoning config triggers info log."""
        mock_result = _successful_result()

        with patch("subtitle_translator.queue.worker.BatchProcessor") as MockBP:
            processor = AsyncMock()
            processor.process_all_batches = AsyncMock(return_value=mock_result)
            MockBP.return_value = processor

            with patch("subtitle_translator.queue.worker.map_translations_to_lines") as mock_map:
                mock_map.return_value = [SubtitleLine(position=1, line="Hola")]

                job_id = await manager.submit_job(
                    request_data={
                        "sourceLanguage": "en",
                        "targetLanguage": "es",
                        "lines": [{"position": 1, "line": "Hello"}],
                        "config": {
                            "reasoning": {"enabled": True, "effort": "medium"},
                        },
                    },
                    job_type=JobType.TRANSLATE_CONTENT,
                )
                manager.set_job_processing(job_id)

                with patch("subtitle_translator.queue.worker.logger") as mock_logger:
                    await process_content_translation_job(manager, job_id, mock_translator)

                    info_calls = [str(c) for c in mock_logger.info.call_args_list]
                    reasoning_log = [c for c in info_calls if "Reasoning config" in c]
                    assert len(reasoning_log) == 1

                job = manager.get_job(job_id)
                assert job.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_provider_config_logged(self, manager, mock_translator):
        """Line 87: provider config triggers info log."""
        mock_result = _successful_result()

        with patch("subtitle_translator.queue.worker.BatchProcessor") as MockBP:
            processor = AsyncMock()
            processor.process_all_batches = AsyncMock(return_value=mock_result)
            MockBP.return_value = processor

            with patch("subtitle_translator.queue.worker.map_translations_to_lines") as mock_map:
                mock_map.return_value = [SubtitleLine(position=1, line="Hola")]

                job_id = await manager.submit_job(
                    request_data={
                        "sourceLanguage": "en",
                        "targetLanguage": "es",
                        "lines": [{"position": 1, "line": "Hello"}],
                        "config": {
                            "provider": {"order": ["deepinfra"], "allowFallbacks": False},
                        },
                    },
                    job_type=JobType.TRANSLATE_CONTENT,
                )
                manager.set_job_processing(job_id)

                with patch("subtitle_translator.queue.worker.logger") as mock_logger:
                    await process_content_translation_job(manager, job_id, mock_translator)

                    info_calls = [str(c) for c in mock_logger.info.call_args_list]
                    provider_log = [c for c in info_calls if "Provider config" in c]
                    assert len(provider_log) == 1

                job = manager.get_job(job_id)
                assert job.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_api_key_override_sets_on_existing_config_without_key(
        self, manager, mock_translator
    ):
        """Lines 71-72: config_override exists but has no api_key, so job.api_key_override fills it."""
        mock_result = _successful_result()

        with patch("subtitle_translator.queue.worker.BatchProcessor") as MockBP:
            processor = AsyncMock()
            processor.process_all_batches = AsyncMock(return_value=mock_result)
            MockBP.return_value = processor

            with patch("subtitle_translator.queue.worker.map_translations_to_lines") as mock_map:
                mock_map.return_value = [SubtitleLine(position=1, line="Hola")]

                job_id = await manager.submit_job(
                    request_data={
                        "sourceLanguage": "en",
                        "targetLanguage": "es",
                        "lines": [{"position": 1, "line": "Hello"}],
                        "config": {"model": "gpt-4"},
                    },
                    job_type=JobType.TRANSLATE_CONTENT,
                    api_key_override="sk-override-key",
                )
                manager.set_job_processing(job_id)
                await process_content_translation_job(manager, job_id, mock_translator)

                call_kwargs = processor.process_all_batches.call_args.kwargs
                config = call_kwargs.get("config_override")
                assert config is not None
                assert config.api_key == "sk-override-key"
                assert config.model == "gpt-4"


# ---------------------------------------------------------------------------
# Content job: progress callback (lines 97-103)
# ---------------------------------------------------------------------------


class TestContentJobProgressCallback:
    """Tests that the progress callback updates the job manager."""

    @pytest.mark.asyncio
    async def test_progress_callback_updates_manager(self, manager, mock_translator):
        """Lines 97-112: progress_callback computes message and calls update_progress."""
        mock_result = _successful_result()

        with patch("subtitle_translator.queue.worker.BatchProcessor") as MockBP:
            processor = AsyncMock()
            processor.process_all_batches = AsyncMock(return_value=mock_result)
            MockBP.return_value = processor

            with patch("subtitle_translator.queue.worker.map_translations_to_lines") as mock_map:
                mock_map.return_value = [SubtitleLine(position=1, line="Hola")]

                job_id = await manager.submit_job(
                    request_data={
                        "sourceLanguage": "en",
                        "targetLanguage": "es",
                        "lines": [{"position": 1, "line": "Hello"}],
                    },
                    job_type=JobType.TRANSLATE_CONTENT,
                )
                manager.set_job_processing(job_id)
                await process_content_translation_job(manager, job_id, mock_translator)

                # Grab the progress_callback passed to process_all_batches
                call_kwargs = processor.process_all_batches.call_args.kwargs
                progress_cb = call_kwargs["progress_callback"]

                # Simulate a progress update with failed batches
                progress = BatchProgress(
                    total_batches=4,
                    completed_batches=2,
                    total_lines=100,
                    completed_lines=50,
                    failed_batches=1,
                    total_tokens=500,
                    total_cost=0.02,
                )
                progress_cb(progress)

                job = manager.get_job(job_id)
                assert job.progress == 50
                assert "50/100 lines" in job.message
                assert "1 failed" in job.message

    @pytest.mark.asyncio
    async def test_progress_callback_without_failures(self, manager, mock_translator):
        """Progress callback message omits failed info when no failures."""
        mock_result = _successful_result()

        with patch("subtitle_translator.queue.worker.BatchProcessor") as MockBP:
            processor = AsyncMock()
            processor.process_all_batches = AsyncMock(return_value=mock_result)
            MockBP.return_value = processor

            with patch("subtitle_translator.queue.worker.map_translations_to_lines") as mock_map:
                mock_map.return_value = [SubtitleLine(position=1, line="Hola")]

                job_id = await manager.submit_job(
                    request_data={
                        "sourceLanguage": "en",
                        "targetLanguage": "es",
                        "lines": [{"position": 1, "line": "Hello"}],
                    },
                    job_type=JobType.TRANSLATE_CONTENT,
                )
                manager.set_job_processing(job_id)
                await process_content_translation_job(manager, job_id, mock_translator)

                call_kwargs = processor.process_all_batches.call_args.kwargs
                progress_cb = call_kwargs["progress_callback"]

                progress = BatchProgress(
                    total_batches=2,
                    completed_batches=1,
                    total_lines=50,
                    completed_lines=25,
                    failed_batches=0,
                    total_tokens=200,
                    total_cost=0.01,
                )
                progress_cb(progress)

                job = manager.get_job(job_id)
                assert "failed" not in job.message


# ---------------------------------------------------------------------------
# Content job: partial failure (lines 133-141)
# ---------------------------------------------------------------------------


class TestContentJobPartialFailure:
    """Tests for partial translation results on content jobs."""

    @pytest.mark.asyncio
    async def test_partial_failure_sets_partial_status(self, manager, mock_translator):
        """Lines 133-150: partial results lead to PARTIAL status with correct error."""
        mock_result = _partial_result(
            translations=[{"index": "0", "content": "Hola"}],
            tokens=80,
        )

        with patch("subtitle_translator.queue.worker.BatchProcessor") as MockBP:
            processor = AsyncMock()
            processor.process_all_batches = AsyncMock(return_value=mock_result)
            MockBP.return_value = processor

            with patch("subtitle_translator.queue.worker.map_translations_to_lines") as mock_map:
                mock_map.return_value = [SubtitleLine(position=1, line="Hola")]

                job_id = await manager.submit_job(
                    request_data={
                        "sourceLanguage": "en",
                        "targetLanguage": "es",
                        "lines": [
                            {"position": 1, "line": "Hello"},
                            {"position": 2, "line": "World"},
                        ],
                    },
                    job_type=JobType.TRANSLATE_CONTENT,
                )
                manager.set_job_processing(job_id)
                await process_content_translation_job(manager, job_id, mock_translator)

                job = manager.get_job(job_id)
                assert job.status == JobStatus.PARTIAL
                assert "1/2 lines translated" in job.error
                assert "1 of 2 batches failed" in job.error
                assert job.result["tokens_used"] == 80
                assert job.result["model_used"] == "test-model"


# ---------------------------------------------------------------------------
# File job: config override and logging (lines 222-236)
# ---------------------------------------------------------------------------


class TestFileJobConfigOverride:
    """Tests for file translation job config override paths."""

    def _setup_srt_mocks(self, mock_translator):
        entries = [MagicMock()]
        mock_translator._srt_parser.parse.return_value = entries
        mock_translator._srt_parser.extract_lines_for_translation.return_value = [
            {"index": "0", "content": "Hello"}
        ]
        mock_translator._srt_parser.apply_translations.return_value = entries
        mock_translator._srt_parser.split_long_subtitles.return_value = entries
        mock_translator._srt_parser.compose.return_value = (
            "1\n00:00:01,000 --> 00:00:02,000\nHola\n\n"
        )
        return entries

    @pytest.mark.asyncio
    async def test_file_job_api_key_override_no_existing_config(self, manager, mock_translator):
        """Lines 222-223: api_key_override creates new TranslationConfig when config_override is None."""
        self._setup_srt_mocks(mock_translator)
        mock_result = _successful_result()

        with patch("subtitle_translator.queue.worker.BatchProcessor") as MockBP:
            processor = AsyncMock()
            processor.process_all_batches = AsyncMock(return_value=mock_result)
            MockBP.return_value = processor

            job_id = await manager.submit_job(
                request_data={
                    "content": "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n",
                    "sourceLanguage": "en",
                    "targetLanguage": "es",
                },
                job_type=JobType.TRANSLATE_FILE,
                api_key_override="sk-file-key",
            )
            manager.set_job_processing(job_id)
            await process_file_translation_job(manager, job_id, mock_translator)

            call_kwargs = processor.process_all_batches.call_args.kwargs
            config = call_kwargs.get("config_override")
            assert config is not None
            assert config.api_key == "sk-file-key"

            job = manager.get_job(job_id)
            assert job.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_file_job_api_key_override_existing_config_no_key(self, manager, mock_translator):
        """Lines 224-225: api_key_override fills existing config that lacks api_key."""
        self._setup_srt_mocks(mock_translator)
        mock_result = _successful_result()

        with patch("subtitle_translator.queue.worker.BatchProcessor") as MockBP:
            processor = AsyncMock()
            processor.process_all_batches = AsyncMock(return_value=mock_result)
            MockBP.return_value = processor

            job_id = await manager.submit_job(
                request_data={
                    "content": "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n",
                    "sourceLanguage": "en",
                    "targetLanguage": "es",
                    "config": {"model": "gpt-4"},
                },
                job_type=JobType.TRANSLATE_FILE,
                api_key_override="sk-file-key-2",
            )
            manager.set_job_processing(job_id)
            await process_file_translation_job(manager, job_id, mock_translator)

            call_kwargs = processor.process_all_batches.call_args.kwargs
            config = call_kwargs.get("config_override")
            assert config is not None
            assert config.api_key == "sk-file-key-2"

    @pytest.mark.asyncio
    async def test_file_job_config_logging_masks_sensitive(self, manager, mock_translator):
        """Lines 229-236: file job config logging masks sensitive keys."""
        self._setup_srt_mocks(mock_translator)
        mock_result = _successful_result()

        with patch("subtitle_translator.queue.worker.BatchProcessor") as MockBP:
            processor = AsyncMock()
            processor.process_all_batches = AsyncMock(return_value=mock_result)
            MockBP.return_value = processor

            job_id = await manager.submit_job(
                request_data={
                    "content": "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n",
                    "sourceLanguage": "en",
                    "targetLanguage": "es",
                    "config": {
                        "apiKey": "sk-top-secret",
                        "model": "gpt-4",
                    },
                },
                job_type=JobType.TRANSLATE_FILE,
            )
            manager.set_job_processing(job_id)

            with patch("subtitle_translator.queue.worker.logger") as mock_logger:
                await process_file_translation_job(manager, job_id, mock_translator)

                info_calls = [str(c) for c in mock_logger.info.call_args_list]
                config_log = [c for c in info_calls if "File translation config" in c]
                assert len(config_log) == 1
                assert "***" in config_log[0]
                assert "sk-top-secret" not in config_log[0]

    @pytest.mark.asyncio
    async def test_file_job_per_request_api_key_debug_log(self, manager, mock_translator):
        """Line 240: per-request API key debug log for file jobs."""
        self._setup_srt_mocks(mock_translator)
        mock_result = _successful_result()

        with patch("subtitle_translator.queue.worker.BatchProcessor") as MockBP:
            processor = AsyncMock()
            processor.process_all_batches = AsyncMock(return_value=mock_result)
            MockBP.return_value = processor

            job_id = await manager.submit_job(
                request_data={
                    "content": "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n",
                    "sourceLanguage": "en",
                    "targetLanguage": "es",
                },
                job_type=JobType.TRANSLATE_FILE,
                api_key_override="sk-debug-key",
            )
            manager.set_job_processing(job_id)

            with patch("subtitle_translator.queue.worker.logger") as mock_logger:
                await process_file_translation_job(manager, job_id, mock_translator)

                debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
                key_log = [c for c in debug_calls if "per-request API key" in c]
                assert len(key_log) == 1


# ---------------------------------------------------------------------------
# File job: RTL language (line 312 in partial path)
# ---------------------------------------------------------------------------


class TestFileJobRTL:
    """Tests for RTL language handling in file translation."""

    def _setup_srt_mocks(self, mock_translator):
        entries = [MagicMock()]
        mock_translator._srt_parser.parse.return_value = entries
        mock_translator._srt_parser.extract_lines_for_translation.return_value = [
            {"index": "0", "content": "Hello"}
        ]
        mock_translator._srt_parser.apply_translations.return_value = entries
        mock_translator._srt_parser.split_long_subtitles.return_value = entries
        mock_translator._srt_parser.compose.return_value = "translated content"
        return entries

    @pytest.mark.asyncio
    async def test_file_job_rtl_language(self, manager, mock_translator):
        """Successful file job with RTL language passes is_rtl=True."""
        self._setup_srt_mocks(mock_translator)
        mock_translator.settings.is_rtl_language = MagicMock(return_value=True)
        mock_result = _successful_result()

        with patch("subtitle_translator.queue.worker.BatchProcessor") as MockBP:
            processor = AsyncMock()
            processor.process_all_batches = AsyncMock(return_value=mock_result)
            MockBP.return_value = processor

            job_id = await manager.submit_job(
                request_data={
                    "content": "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n",
                    "sourceLanguage": "en",
                    "targetLanguage": "ar",
                },
                job_type=JobType.TRANSLATE_FILE,
            )
            manager.set_job_processing(job_id)
            await process_file_translation_job(manager, job_id, mock_translator)

            mock_translator._srt_parser.apply_translations.assert_called_once()
            call_kwargs = mock_translator._srt_parser.apply_translations.call_args
            assert call_kwargs.kwargs.get("is_rtl") is True or call_kwargs[1].get("is_rtl") is True

            job = manager.get_job(job_id)
            assert job.status == JobStatus.COMPLETED


# ---------------------------------------------------------------------------
# File job: partial failure (lines 308-336)
# ---------------------------------------------------------------------------


class TestFileJobPartialFailure:
    """Tests for partial failure in file translation jobs."""

    def _setup_srt_mocks(self, mock_translator):
        entries = [MagicMock()]
        mock_translator._srt_parser.parse.return_value = entries
        mock_translator._srt_parser.extract_lines_for_translation.return_value = [
            {"index": "0", "content": "Hello"},
            {"index": "1", "content": "World"},
        ]
        mock_translator._srt_parser.apply_translations.return_value = entries
        mock_translator._srt_parser.compose.return_value = "partial translated content"
        return entries

    @pytest.mark.asyncio
    async def test_file_partial_failure(self, manager, mock_translator):
        """Lines 308-330: partial file translation sets PARTIAL status."""
        self._setup_srt_mocks(mock_translator)
        mock_result = _partial_result(
            translations=[{"index": "0", "content": "Hola"}],
            tokens=60,
        )

        with patch("subtitle_translator.queue.worker.BatchProcessor") as MockBP:
            processor = AsyncMock()
            processor.process_all_batches = AsyncMock(return_value=mock_result)
            MockBP.return_value = processor

            job_id = await manager.submit_job(
                request_data={
                    "content": "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n",
                    "sourceLanguage": "en",
                    "targetLanguage": "es",
                },
                job_type=JobType.TRANSLATE_FILE,
            )
            manager.set_job_processing(job_id)
            await process_file_translation_job(manager, job_id, mock_translator)

            job = manager.get_job(job_id)
            assert job.status == JobStatus.PARTIAL
            assert "1/2 lines translated" in job.error
            assert "1 of 2 batches failed" in job.error
            assert job.result["content"] == "partial translated content"
            assert job.result["tokens_used"] == 60

    @pytest.mark.asyncio
    async def test_file_all_batches_failed(self, manager, mock_translator):
        """Lines 331-335: all batches failed sets FAILED status."""
        entries = [MagicMock()]
        mock_translator._srt_parser.parse.return_value = entries
        mock_translator._srt_parser.extract_lines_for_translation.return_value = [
            {"index": "0", "content": "Hello"}
        ]
        mock_result = _all_failed_result()

        with patch("subtitle_translator.queue.worker.BatchProcessor") as MockBP:
            processor = AsyncMock()
            processor.process_all_batches = AsyncMock(return_value=mock_result)
            MockBP.return_value = processor

            job_id = await manager.submit_job(
                request_data={
                    "content": "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n",
                    "sourceLanguage": "en",
                    "targetLanguage": "es",
                },
                job_type=JobType.TRANSLATE_FILE,
            )
            manager.set_job_processing(job_id)
            await process_file_translation_job(manager, job_id, mock_translator)

            job = manager.get_job(job_id)
            assert job.status == JobStatus.FAILED
            assert "All 1 batches failed" in job.error


# ---------------------------------------------------------------------------
# File job: progress callback (lines 278-284)
# ---------------------------------------------------------------------------


class TestFileJobProgressCallback:
    """Tests for file job progress callback."""

    def _setup_srt_mocks(self, mock_translator):
        entries = [MagicMock()]
        mock_translator._srt_parser.parse.return_value = entries
        mock_translator._srt_parser.extract_lines_for_translation.return_value = [
            {"index": "0", "content": "Hello"}
        ]
        mock_translator._srt_parser.apply_translations.return_value = entries
        mock_translator._srt_parser.split_long_subtitles.return_value = entries
        mock_translator._srt_parser.compose.return_value = "translated"
        return entries

    @pytest.mark.asyncio
    async def test_file_progress_callback(self, manager, mock_translator):
        """Lines 278-293: file job progress callback updates job manager."""
        self._setup_srt_mocks(mock_translator)
        mock_result = _successful_result()

        with patch("subtitle_translator.queue.worker.BatchProcessor") as MockBP:
            processor = AsyncMock()
            processor.process_all_batches = AsyncMock(return_value=mock_result)
            MockBP.return_value = processor

            job_id = await manager.submit_job(
                request_data={
                    "content": "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n",
                    "sourceLanguage": "en",
                    "targetLanguage": "es",
                },
                job_type=JobType.TRANSLATE_FILE,
            )
            manager.set_job_processing(job_id)
            await process_file_translation_job(manager, job_id, mock_translator)

            call_kwargs = processor.process_all_batches.call_args.kwargs
            progress_cb = call_kwargs["progress_callback"]

            # Fire the callback manually
            progress = BatchProgress(
                total_batches=3,
                completed_batches=1,
                total_lines=60,
                completed_lines=20,
                failed_batches=0,
                total_tokens=300,
                total_cost=0.01,
            )
            progress_cb(progress)

            job = manager.get_job(job_id)
            assert job.progress == 33  # int(1/3 * 100)
            assert "20/60 lines" in job.message

    @pytest.mark.asyncio
    async def test_file_progress_callback_with_failures(self, manager, mock_translator):
        """File job progress callback includes failed info when present."""
        self._setup_srt_mocks(mock_translator)
        mock_result = _successful_result()

        with patch("subtitle_translator.queue.worker.BatchProcessor") as MockBP:
            processor = AsyncMock()
            processor.process_all_batches = AsyncMock(return_value=mock_result)
            MockBP.return_value = processor

            job_id = await manager.submit_job(
                request_data={
                    "content": "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n",
                    "sourceLanguage": "en",
                    "targetLanguage": "es",
                },
                job_type=JobType.TRANSLATE_FILE,
            )
            manager.set_job_processing(job_id)
            await process_file_translation_job(manager, job_id, mock_translator)

            call_kwargs = processor.process_all_batches.call_args.kwargs
            progress_cb = call_kwargs["progress_callback"]

            progress = BatchProgress(
                total_batches=4,
                completed_batches=3,
                total_lines=80,
                completed_lines=60,
                failed_batches=2,
                total_tokens=600,
                total_cost=0.03,
            )
            progress_cb(progress)

            job = manager.get_job(job_id)
            assert "2 failed" in job.message


# ---------------------------------------------------------------------------
# File job: split_long_subtitles (called on success path)
# ---------------------------------------------------------------------------


class TestFileJobSplitLongSubtitles:
    """Verify split_long_subtitles is invoked on the success path."""

    @pytest.mark.asyncio
    async def test_split_long_subtitles_called(self, manager, mock_translator):
        entries = [MagicMock()]
        split_entries = [MagicMock(), MagicMock()]
        mock_translator._srt_parser.parse.return_value = entries
        mock_translator._srt_parser.extract_lines_for_translation.return_value = [
            {"index": "0", "content": "Hello"}
        ]
        mock_translator._srt_parser.apply_translations.return_value = entries
        mock_translator._srt_parser.split_long_subtitles.return_value = split_entries
        mock_translator._srt_parser.compose.return_value = "split output"

        mock_result = _successful_result()

        with patch("subtitle_translator.queue.worker.BatchProcessor") as MockBP:
            processor = AsyncMock()
            processor.process_all_batches = AsyncMock(return_value=mock_result)
            MockBP.return_value = processor

            job_id = await manager.submit_job(
                request_data={
                    "content": "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n",
                    "sourceLanguage": "en",
                    "targetLanguage": "es",
                },
                job_type=JobType.TRANSLATE_FILE,
            )
            manager.set_job_processing(job_id)
            await process_file_translation_job(manager, job_id, mock_translator)

            mock_translator._srt_parser.split_long_subtitles.assert_called_once_with(entries)
            # compose receives the split entries
            mock_translator._srt_parser.compose.assert_called_once_with(split_entries)

            job = manager.get_job(job_id)
            assert job.status == JobStatus.COMPLETED
            assert job.result["content"] == "split output"


# ---------------------------------------------------------------------------
# File job: exception handler (lines 362-364)
# ---------------------------------------------------------------------------


class TestFileJobExceptionHandler:
    """Tests for the outer exception handler in file translation."""

    @pytest.mark.asyncio
    async def test_unexpected_exception_sets_failed(self, manager, mock_translator):
        """Lines 362-364: unexpected exception caught by outer handler."""
        # Make extract_lines_for_translation raise after parse succeeds
        mock_translator._srt_parser.parse.return_value = [MagicMock()]
        mock_translator._srt_parser.extract_lines_for_translation.side_effect = ValueError(
            "unexpected format"
        )

        job_id = await manager.submit_job(
            request_data={
                "content": "valid srt content",
                "sourceLanguage": "en",
                "targetLanguage": "es",
            },
            job_type=JobType.TRANSLATE_FILE,
        )
        manager.set_job_processing(job_id)
        await process_file_translation_job(manager, job_id, mock_translator)

        job = manager.get_job(job_id)
        assert job.status == JobStatus.FAILED
        assert "unexpected format" in job.error


# ---------------------------------------------------------------------------
# Config extraction: parallelBatches key
# ---------------------------------------------------------------------------


class TestExtractConfigParallelBatches:
    """Test that parallelBatches is recognized by _extract_config_override_from_dict."""

    def test_parallel_batches_alias_not_in_check_list(self):
        """parallelBatches is not in the recognized key list, so dict with only that key returns None."""
        result = _extract_config_override_from_dict({"parallelBatches": 3})
        assert result is None

    def test_parallel_batches_with_recognized_key(self):
        """parallelBatches works when combined with a recognized key."""
        result = _extract_config_override_from_dict({"model": "gpt-4", "parallelBatches": 3})
        assert result is not None
        assert result.parallel_batches == 3
        assert result.model == "gpt-4"
