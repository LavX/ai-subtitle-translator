"""Tests for the batch activity a job reports while it is running.

The activity strings are the only account of a running job a user ever sees,
so they have to survive the adaptive recovery path. A batch that shrinks after
a timeout keeps working on the same batch, and the report has to say so: a
recovery attempt that presents itself as the first attempt hides a batch that
has already spent several full request budgets.
"""

import logging

import pytest

from subtitle_translator.core.batch_processor import BatchProcessor
from subtitle_translator.core.batch_sizing import get_batch_size_resolver
from subtitle_translator.providers.base import (
    ProviderTimeoutError,
    TranslationBatch,
    TranslationResult,
)


class TestBatchActivityReporting:
    """The reported attempt number and batch number must stay truthful."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from unittest.mock import MagicMock

        self.provider = MagicMock()
        self.provider.get_model_metadata.return_value = None
        self.settings = MagicMock()
        self.settings.request_timeout = 120.0
        self.settings.batch_size = 100
        self.settings.max_retries = 2
        self.settings.retry_delay = 0.01
        self.settings.openrouter_default_model = "test/model"
        self.processor = BatchProcessor(self.provider, self.settings)
        get_batch_size_resolver().reset()
        get_batch_size_resolver()._settings = self.settings

    def _timeout_over(self, limit):
        """Time out above `limit` lines, translate at or below it."""

        async def mock_translate(batch, **kwargs):
            if len(batch.lines) > limit:
                raise ProviderTimeoutError("Request timeout")
            return TranslationResult(
                translations=[
                    {"index": line["index"], "content": f"T-{line['content']}"}
                    for line in batch.lines
                ],
                model_used="test/model",
                total_tokens=10,
            )

        return mock_translate

    @pytest.mark.asyncio
    async def test_attempt_number_accumulates_across_adaptive_recovery(self):
        """A recovery request is not attempt 1: it follows the attempt that timed out."""
        lines = [{"index": str(i), "content": f"Line {i}"} for i in range(10)]
        self.provider.translate_batch = self._timeout_over(5)
        activity = []

        batch = TranslationBatch(lines=lines, source_language="en", target_language="hu")
        result = await self.processor.process_batch(
            batch,
            batch_index=0,
            model="test/model",
            _activity_callback=activity.append,
        )

        assert result.success is True
        in_progress = [m for m in activity if "request in progress" in m]
        assert in_progress[0] == "Batch 1: request in progress for 10 lines (attempt 1)"
        recovery = [m for m in in_progress if "for 5 lines" in m]
        assert recovery, f"expected a 5-line recovery request, got {in_progress}"
        assert all("(attempt 1)" not in m for m in recovery), (
            "a recovery request after a timeout reported itself as the first attempt: "
            f"{recovery}"
        )

    @pytest.mark.asyncio
    async def test_batch_number_is_one_based_in_logs_and_activity(self, caplog):
        """One batch must carry one number, whether it is logged or reported."""
        lines = [{"index": str(i), "content": f"Line {i}"} for i in range(10)]
        self.provider.translate_batch = self._timeout_over(5)
        activity = []

        batch = TranslationBatch(lines=lines, source_language="en", target_language="hu")
        with caplog.at_level(logging.WARNING, logger="subtitle_translator.core.batch_processor"):
            result = await self.processor.process_batch(
                batch,
                batch_index=3,
                model="test/model",
                _activity_callback=activity.append,
            )

        assert result.success is True
        assert all(m.startswith("Batch 4:") for m in activity), activity
        retry_logs = [r.message for r in caplog.records if "adaptive retry with size" in r.message]
        assert retry_logs, "expected an adaptive retry log record"
        assert all(
            m.startswith("Batch 4:") for m in retry_logs
        ), f"log numbers the batch differently from the activity stream: {retry_logs}"

    @pytest.mark.asyncio
    async def test_every_reported_activity_reaches_the_browser(self):
        """The GUI forwards activity by matching an allowlist of exact strings.

        The allowlist lives in gui.py and the strings are built in this module, so
        nothing but this test stops the two from drifting. Drift is silent: an
        unmatched message is replaced by an empty one, and the browser simply stops
        reporting what the job is doing.
        """
        from subtitle_translator.gui import _ACTIVITY

        lines = [{"index": str(i), "content": f"Line {i}"} for i in range(10)]
        self.provider.translate_batch = self._timeout_over(5)
        activity = []

        batch = TranslationBatch(lines=lines, source_language="en", target_language="hu")
        result = await self.processor.process_batch(
            batch,
            batch_index=0,
            model="test/model",
            _activity_callback=activity.append,
        )

        assert result.success is True
        assert activity, "expected the run to report activity"
        unmatched = [m for m in activity if not _ACTIVITY.fullmatch(m)]
        assert (
            not unmatched
        ), f"the GUI would blank these messages instead of showing them: {unmatched}"


class TestTransportFailureReporting:
    """A transport failure has to name itself in the log."""

    @pytest.mark.asyncio
    async def test_empty_transport_error_still_names_the_failure(self):
        """httpx.ReadError has an empty str(), which logged as 'Network error: '."""
        import httpx

        from subtitle_translator.config import Settings
        from subtitle_translator.providers.base import TranslationProviderError
        from subtitle_translator.providers.openrouter import OpenRouterProvider

        def explode(request):
            raise httpx.ReadError("")

        provider = OpenRouterProvider(
            Settings(_env_file=None, openrouter_api_key="synthetic-test-only")
        )
        provider._client = httpx.AsyncClient(
            transport=httpx.MockTransport(explode), base_url="https://fake"
        )
        provider._model_params_fetched = True

        batch = TranslationBatch(
            lines=[{"index": "0", "content": "Cue"}],
            source_language="en",
            target_language="hu",
        )
        try:
            with pytest.raises(TranslationProviderError) as caught:
                await provider.translate_batch(batch, model="test/model")
        finally:
            await provider._client.aclose()

        assert str(caught.value).rstrip() != "Network error:"
        assert "ReadError" in str(caught.value)
