"""Tests for job queue API routes and _build_job_status_response helper."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from subtitle_translator.api.routes import _build_job_status_response
from subtitle_translator.main import app
from subtitle_translator.queue.job_manager import Job, JobStatus, JobType, job_manager


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_settings():
    with patch("subtitle_translator.api.routes.get_settings") as mock:
        settings = MagicMock()
        settings.openrouter_api_key = "test-api-key"
        settings.openrouter_default_model = "google/gemini-2.5-flash-preview-09-2025"
        settings.openrouter_temperature = 0.3
        settings.batch_size = 25
        settings.parallel_batches_per_job = 4
        settings.admin_api_key = ""
        mock.return_value = settings
        yield settings


@pytest.fixture(autouse=True)
def clean_job_manager():
    """Clear all jobs between tests."""
    job_manager.jobs.clear()
    # Drain the queue
    while not job_manager.queue.empty():
        try:
            job_manager.queue.get_nowait()
        except Exception:
            break
    yield
    job_manager.jobs.clear()


def _make_content_request(**overrides):
    base = {
        "sourceLanguage": "en",
        "targetLanguage": "hu",
        "lines": [{"position": 1, "line": "Hello"}],
    }
    base.update(overrides)
    return base


def _make_file_request(**overrides):
    base = {
        "content": "1\n00:00:01,000 --> 00:00:04,000\nHello\n\n",
        "sourceLanguage": "en",
        "targetLanguage": "hu",
    }
    base.update(overrides)
    return base


# ============================================================================
# _build_job_status_response unit tests
# ============================================================================


class TestBuildJobStatusResponse:
    """Tests for the _build_job_status_response helper."""

    def test_queued_job_no_elapsed(self):
        job = Job(
            id="test-1",
            job_type=JobType.TRANSLATE_CONTENT,
            status=JobStatus.QUEUED,
            request_data={},
            created_at=datetime.now(UTC),
            job_name="my-job",
            file_name="subs.srt",
            source_language="en",
            target_language="hu",
            title="Test Movie",
            media_type="Movie",
            model="test-model",
            total_lines=50,
        )
        resp = _build_job_status_response(job)
        assert resp.jobId == "test-1"
        assert resp.status == "queued"
        assert resp.elapsedSeconds is None
        assert resp.jobName == "my-job"
        assert resp.fileName == "subs.srt"
        assert resp.sourceLanguage == "en"
        assert resp.targetLanguage == "hu"
        assert resp.title == "Test Movie"
        assert resp.mediaType == "Movie"
        assert resp.model == "test-model"
        assert resp.totalLines == 50

    def test_processing_job_elapsed_computed(self):
        started = datetime.now(UTC) - timedelta(seconds=30)
        job = Job(
            id="test-2",
            job_type=JobType.TRANSLATE_FILE,
            status=JobStatus.PROCESSING,
            request_data={},
            created_at=datetime.now(UTC),
            started_at=started,
            total_batches=10,
            completed_batches=5,
            completed_lines=100,
            tokens_used=5000,
            total_cost=0.05,
        )
        resp = _build_job_status_response(job)
        assert resp.elapsedSeconds is not None
        assert resp.elapsedSeconds >= 29  # at least ~30 seconds
        assert resp.totalBatches == 10
        assert resp.completedBatches == 5
        assert resp.completedLines == 100
        assert resp.tokensUsed == 5000
        assert resp.totalCost == pytest.approx(0.05)

    def test_completed_job_elapsed_fixed(self):
        started = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        completed = datetime(2026, 1, 1, 0, 0, 45, tzinfo=UTC)
        job = Job(
            id="test-3",
            job_type=JobType.TRANSLATE_CONTENT,
            status=JobStatus.COMPLETED,
            request_data={},
            created_at=started,
            started_at=started,
            completed_at=completed,
            result={"lines": []},
        )
        resp = _build_job_status_response(job)
        assert resp.elapsedSeconds == 45.0
        assert resp.result == {"lines": []}

    def test_failed_job_shows_error(self):
        job = Job(
            id="test-4",
            job_type=JobType.TRANSLATE_CONTENT,
            status=JobStatus.FAILED,
            request_data={},
            created_at=datetime.now(UTC),
            error="Translation failed",
        )
        resp = _build_job_status_response(job)
        assert resp.error == "Translation failed"
        assert resp.result is None

    def test_zero_tokens_and_cost_shown_as_none(self):
        job = Job(
            id="test-5",
            job_type=JobType.TRANSLATE_CONTENT,
            status=JobStatus.PROCESSING,
            request_data={},
            created_at=datetime.now(UTC),
            tokens_used=0,
            total_cost=0.0,
        )
        resp = _build_job_status_response(job)
        assert resp.tokensUsed is None
        assert resp.totalCost is None

    def test_no_total_batches_hides_batch_metrics(self):
        job = Job(
            id="test-6",
            job_type=JobType.TRANSLATE_CONTENT,
            status=JobStatus.QUEUED,
            request_data={},
            created_at=datetime.now(UTC),
            total_batches=None,
            completed_batches=0,
            completed_lines=0,
        )
        resp = _build_job_status_response(job)
        assert resp.totalBatches is None
        assert resp.completedBatches is None
        assert resp.completedLines is None

    def test_metadata_defaults_to_none(self):
        job = Job(
            id="test-7",
            job_type=JobType.TRANSLATE_CONTENT,
            status=JobStatus.QUEUED,
            request_data={},
            created_at=datetime.now(UTC),
        )
        resp = _build_job_status_response(job)
        assert resp.jobName is None
        assert resp.fileName is None
        assert resp.sourceLanguage is None
        assert resp.model is None


# ============================================================================
# Submit content translation job
# ============================================================================


class TestSubmitContentJob:
    """Tests for POST /api/v1/jobs/translate/content."""

    def test_submit_content_job_success(self, client, mock_settings):
        resp = client.post(
            "/api/v1/jobs/translate/content",
            json=_make_content_request(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "jobId" in data
        assert data["status"] == "queued"
        assert data["position"] == 1

    def test_submit_content_job_with_metadata(self, client, mock_settings):
        resp = client.post(
            "/api/v1/jobs/translate/content",
            json=_make_content_request(
                title="Breaking Bad",
                mediaType="Episode",
                fileName="breaking_bad_s01e01.srt",
                jobName="bb-translate",
            ),
        )
        assert resp.status_code == 200
        job_id = resp.json()["jobId"]

        # Verify metadata stored on job
        job = job_manager.get_job(job_id)
        assert job.job_name == "bb-translate"
        assert job.file_name == "breaking_bad_s01e01.srt"
        assert job.source_language == "en"
        assert job.target_language == "hu"
        assert job.title == "Breaking Bad"
        assert job.media_type == "Episode"
        assert job.total_lines == 1

    def test_submit_content_job_model_from_config(self, client, mock_settings):
        resp = client.post(
            "/api/v1/jobs/translate/content",
            json=_make_content_request(
                config={"model": "anthropic/claude-haiku-4.5"},
            ),
        )
        assert resp.status_code == 200
        job_id = resp.json()["jobId"]
        job = job_manager.get_job(job_id)
        assert job.model == "anthropic/claude-haiku-4.5"

    def test_submit_content_job_model_from_request(self, client, mock_settings):
        resp = client.post(
            "/api/v1/jobs/translate/content",
            json=_make_content_request(model="meta-llama/llama-4-maverick"),
        )
        assert resp.status_code == 200
        job_id = resp.json()["jobId"]
        job = job_manager.get_job(job_id)
        assert job.model == "meta-llama/llama-4-maverick"

    def test_submit_content_job_model_defaults(self, client, mock_settings):
        resp = client.post(
            "/api/v1/jobs/translate/content",
            json=_make_content_request(),
        )
        assert resp.status_code == 200
        job_id = resp.json()["jobId"]
        job = job_manager.get_job(job_id)
        assert job.model == "google/gemini-2.5-flash-preview-09-2025"

    def test_submit_content_job_no_api_key(self, client):
        with patch("subtitle_translator.api.routes.get_settings") as mock:
            settings = MagicMock()
            settings.openrouter_api_key = ""
            mock.return_value = settings
            resp = client.post(
                "/api/v1/jobs/translate/content",
                json=_make_content_request(),
            )
            assert resp.status_code == 401

    def test_submit_content_job_with_config_api_key(self, client):
        with patch("subtitle_translator.api.routes.get_settings") as mock:
            settings = MagicMock()
            settings.openrouter_api_key = ""
            settings.openrouter_default_model = "test-model"
            mock.return_value = settings
            resp = client.post(
                "/api/v1/jobs/translate/content",
                json=_make_content_request(
                    config={"apiKey": "sk-from-request"},
                ),
            )
            assert resp.status_code == 200

    def test_submit_content_job_extracts_api_key_to_override(self, client, mock_settings):
        resp = client.post(
            "/api/v1/jobs/translate/content",
            json=_make_content_request(
                config={"apiKey": "sk-override-key"},
            ),
        )
        assert resp.status_code == 200
        job_id = resp.json()["jobId"]
        job = job_manager.get_job(job_id)
        assert job.api_key_override == "sk-override-key"
        # api_key should be removed from request_data config
        config = job.request_data.get("config", {})
        assert config.get("api_key") is None

    def test_submit_content_job_queue_full(self, client, mock_settings):
        with patch.object(
            job_manager, "submit_job", side_effect=RuntimeError("Maximum job limit (100) reached")
        ):
            resp = client.post(
                "/api/v1/jobs/translate/content",
                json=_make_content_request(),
            )
            assert resp.status_code == 429
            assert "queue_full" in resp.json()["detail"]["error"]


# ============================================================================
# Submit file translation job
# ============================================================================


class TestSubmitFileJob:
    """Tests for POST /api/v1/jobs/translate/file."""

    def test_submit_file_job_success(self, client, mock_settings):
        resp = client.post(
            "/api/v1/jobs/translate/file",
            json=_make_file_request(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "jobId" in data
        assert data["status"] == "queued"

    def test_submit_file_job_with_metadata(self, client, mock_settings):
        resp = client.post(
            "/api/v1/jobs/translate/file",
            json=_make_file_request(
                title="Inception",
                mediaType="Movie",
                fileName="inception.srt",
                jobName="inception-job",
            ),
        )
        assert resp.status_code == 200
        job_id = resp.json()["jobId"]
        job = job_manager.get_job(job_id)
        assert job.job_name == "inception-job"
        assert job.file_name == "inception.srt"
        assert job.title == "Inception"
        assert job.media_type == "Movie"
        assert job.total_lines is None  # unknown until parsing

    def test_submit_file_job_empty_content(self, client, mock_settings):
        resp = client.post(
            "/api/v1/jobs/translate/file",
            json=_make_file_request(content=""),
        )
        assert resp.status_code == 400

    def test_submit_file_job_whitespace_content(self, client, mock_settings):
        resp = client.post(
            "/api/v1/jobs/translate/file",
            json=_make_file_request(content="   "),
        )
        assert resp.status_code == 400

    def test_submit_file_job_no_api_key(self, client):
        with patch("subtitle_translator.api.routes.get_settings") as mock:
            settings = MagicMock()
            settings.openrouter_api_key = ""
            mock.return_value = settings
            resp = client.post(
                "/api/v1/jobs/translate/file",
                json=_make_file_request(),
            )
            assert resp.status_code == 401

    def test_submit_file_job_model_resolution(self, client, mock_settings):
        resp = client.post(
            "/api/v1/jobs/translate/file",
            json=_make_file_request(
                config={"model": "anthropic/claude-haiku-4.5"},
            ),
        )
        assert resp.status_code == 200
        job_id = resp.json()["jobId"]
        job = job_manager.get_job(job_id)
        assert job.model == "anthropic/claude-haiku-4.5"

    def test_submit_file_job_with_config_api_key(self, client):
        with patch("subtitle_translator.api.routes.get_settings") as mock:
            settings = MagicMock()
            settings.openrouter_api_key = ""
            settings.openrouter_default_model = "test-model"
            mock.return_value = settings
            resp = client.post(
                "/api/v1/jobs/translate/file",
                json=_make_file_request(
                    config={"apiKey": "sk-from-request"},
                ),
            )
            assert resp.status_code == 200
            job_id = resp.json()["jobId"]
            job = job_manager.get_job(job_id)
            assert job.api_key_override == "sk-from-request"

    def test_submit_file_job_queue_full(self, client, mock_settings):
        with patch.object(
            job_manager, "submit_job", side_effect=RuntimeError("Maximum job limit (100) reached")
        ):
            resp = client.post(
                "/api/v1/jobs/translate/file",
                json=_make_file_request(),
            )
            assert resp.status_code == 429


# ============================================================================
# Get job status
# ============================================================================


class TestGetJobStatus:
    """Tests for GET /api/v1/jobs/{job_id}."""

    def test_get_job_status(self, client, mock_settings):
        # Submit a job first
        resp = client.post(
            "/api/v1/jobs/translate/content",
            json=_make_content_request(
                jobName="test-job",
                fileName="test.srt",
                title="Test",
                mediaType="Movie",
            ),
        )
        job_id = resp.json()["jobId"]

        resp = client.get(f"/api/v1/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["jobId"] == job_id
        assert data["status"] == "queued"
        assert data["jobName"] == "test-job"
        assert data["fileName"] == "test.srt"
        assert data["sourceLanguage"] == "en"
        assert data["targetLanguage"] == "hu"
        assert data["title"] == "Test"
        assert data["mediaType"] == "Movie"
        assert data["model"] == "google/gemini-2.5-flash-preview-09-2025"
        assert data["totalLines"] == 1
        assert data["elapsedSeconds"] is None

    def test_get_job_status_not_found(self, client):
        resp = client.get("/api/v1/jobs/nonexistent-id")
        assert resp.status_code == 404

    def test_get_job_status_with_metrics(self, client, mock_settings):
        resp = client.post(
            "/api/v1/jobs/translate/content",
            json=_make_content_request(),
        )
        job_id = resp.json()["jobId"]

        # Simulate processing with metrics
        job_manager.get_job(job_id)
        job_manager.set_job_processing(job_id)
        job_manager.update_progress(
            job_id,
            50,
            "Processing batch 5/10",
            total_batches=10,
            completed_batches=5,
            completed_lines=100,
            tokens_used=5000,
            total_cost=0.04,
        )

        resp = client.get(f"/api/v1/jobs/{job_id}")
        data = resp.json()
        assert data["status"] == "processing"
        assert data["progress"] == 50
        assert data["totalBatches"] == 10
        assert data["completedBatches"] == 5
        assert data["completedLines"] == 100
        assert data["tokensUsed"] == 5000
        assert data["totalCost"] == pytest.approx(0.04)
        assert data["elapsedSeconds"] is not None
        assert data["elapsedSeconds"] >= 0


# ============================================================================
# List jobs
# ============================================================================


class TestListJobs:
    """Tests for GET /api/v1/jobs."""

    def test_list_jobs_empty(self, client):
        resp = client.get("/api/v1/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["jobs"] == []
        assert data["total"] == 0

    def test_list_jobs_with_metadata(self, client, mock_settings):
        client.post(
            "/api/v1/jobs/translate/content",
            json=_make_content_request(jobName="job-1"),
        )
        client.post(
            "/api/v1/jobs/translate/content",
            json=_make_content_request(jobName="job-2"),
        )

        resp = client.get("/api/v1/jobs")
        data = resp.json()
        assert data["total"] == 2
        assert len(data["jobs"]) == 2
        names = {j["jobName"] for j in data["jobs"]}
        assert names == {"job-1", "job-2"}

    def test_list_jobs_status_filter(self, client, mock_settings):
        resp1 = client.post(
            "/api/v1/jobs/translate/content",
            json=_make_content_request(),
        )
        client.post(
            "/api/v1/jobs/translate/content",
            json=_make_content_request(),
        )
        job_id1 = resp1.json()["jobId"]
        job_manager.set_job_processing(job_id1)

        resp = client.get("/api/v1/jobs?status=processing")
        data = resp.json()
        assert len(data["jobs"]) == 1
        assert data["jobs"][0]["jobId"] == job_id1

    def test_list_jobs_invalid_status_filter(self, client):
        resp = client.get("/api/v1/jobs?status=invalid")
        assert resp.status_code == 400

    def test_list_jobs_with_limit(self, client, mock_settings):
        for _ in range(3):
            client.post(
                "/api/v1/jobs/translate/content",
                json=_make_content_request(),
            )
        resp = client.get("/api/v1/jobs?limit=2")
        data = resp.json()
        assert len(data["jobs"]) == 2


# ============================================================================
# Cancel/Delete job
# ============================================================================


class TestCancelDeleteJob:
    """Tests for DELETE /api/v1/jobs/{job_id}."""

    def test_cancel_queued_job(self, client, mock_settings):
        resp = client.post(
            "/api/v1/jobs/translate/content",
            json=_make_content_request(),
        )
        job_id = resp.json()["jobId"]

        resp = client.delete(f"/api/v1/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "cancelled"

    def test_delete_completed_job(self, client, mock_settings):
        resp = client.post(
            "/api/v1/jobs/translate/content",
            json=_make_content_request(),
        )
        job_id = resp.json()["jobId"]
        job_manager.set_job_processing(job_id)
        job_manager.set_job_completed(job_id, {})

        resp = client.delete(f"/api/v1/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_cannot_cancel_processing_job(self, client, mock_settings):
        resp = client.post(
            "/api/v1/jobs/translate/content",
            json=_make_content_request(),
        )
        job_id = resp.json()["jobId"]
        job_manager.set_job_processing(job_id)

        resp = client.delete(f"/api/v1/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "processing"
        assert "Cannot cancel" in resp.json()["message"]

    def test_delete_not_found(self, client):
        resp = client.delete("/api/v1/jobs/nonexistent-id")
        assert resp.status_code == 404

    def test_delete_failed_job(self, client, mock_settings):
        resp = client.post(
            "/api/v1/jobs/translate/content",
            json=_make_content_request(),
        )
        job_id = resp.json()["jobId"]
        job_manager.set_job_processing(job_id)
        job_manager.set_job_failed(job_id, "error")

        resp = client.delete(f"/api/v1/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"


# ============================================================================
# Config and status endpoints
# ============================================================================


class TestConfigEndpoints:
    """Tests for config and status endpoints."""

    def test_get_config(self, client, mock_settings):
        resp = client.get("/api/v1/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "google/gemini-2.5-flash-preview-09-2025"
        assert data["apiKeyConfigured"] is True

    def test_get_service_status(self, client, mock_settings):
        with patch("subtitle_translator.api.routes.get_translator") as mock_translator:
            translator = AsyncMock()
            translator.health_check = AsyncMock(return_value=True)
            mock_translator.return_value = translator

            resp = client.get("/api/v1/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["healthy"] is True
            assert data["service"] == "ai-subtitle-translator"

    def test_get_service_status_health_check_fails(self, client, mock_settings):
        with patch("subtitle_translator.api.routes.get_translator") as mock_translator:
            translator = AsyncMock()
            translator.health_check = AsyncMock(side_effect=Exception("down"))
            mock_translator.return_value = translator

            resp = client.get("/api/v1/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["healthy"] is False

    def test_update_config_model(self, client, mock_settings):
        with patch("subtitle_translator.api.routes.update_runtime_config"):
            resp = client.put("/api/v1/config", json={"model": "new-model"})
            assert resp.status_code == 200
            assert resp.json()["status"] == "updated"

    def test_update_config_no_fields(self, client, mock_settings):
        resp = client.put("/api/v1/config", json={})
        assert resp.status_code == 200
        assert resp.json()["status"] == "no_change"

    def test_update_config_forbidden(self, client):
        with patch("subtitle_translator.api.routes.get_settings") as mock:
            settings = MagicMock()
            settings.admin_api_key = "secret"
            mock.return_value = settings
            resp = client.put("/api/v1/config", json={"model": "new"})
            assert resp.status_code == 403

    def test_update_config_with_admin_key(self, client):
        with patch("subtitle_translator.api.routes.get_settings") as mock:
            settings = MagicMock()
            settings.admin_api_key = "secret"
            mock.return_value = settings
            with patch("subtitle_translator.api.routes.update_runtime_config"):
                resp = client.put(
                    "/api/v1/config",
                    json={"model": "new"},
                    headers={"X-Admin-Key": "secret"},
                )
                assert resp.status_code == 200

    def test_update_config_temperature(self, client, mock_settings):
        with patch("subtitle_translator.api.routes.update_runtime_config"):
            resp = client.put("/api/v1/config", json={"temperature": 0.5})
            assert resp.status_code == 200
            assert "temperature" in resp.json()["message"]

    def test_update_config_api_key(self, client, mock_settings):
        with patch("subtitle_translator.api.routes.update_runtime_config"):
            resp = client.put("/api/v1/config", json={"apiKey": "new-key"})
            assert resp.status_code == 200
            assert "apiKey" in resp.json()["message"]

    def test_update_config_parallel_batches(self, client, mock_settings):
        with patch("subtitle_translator.api.routes.update_runtime_config"):
            resp = client.put("/api/v1/config", json={"parallelBatchesPerJob": 6})
            assert resp.status_code == 200
            assert "parallelBatchesPerJob" in resp.json()["message"]

    def test_update_config_max_concurrent_jobs(self, client, mock_settings):
        with patch.object(job_manager, "set_max_concurrent", new_callable=AsyncMock):
            resp = client.put("/api/v1/config", json={"maxConcurrentJobs": 5})
            assert resp.status_code == 200
            assert "maxConcurrentJobs" in resp.json()["message"]

    def test_update_config_value_error(self, client, mock_settings):
        with patch(
            "subtitle_translator.api.routes.update_runtime_config",
            side_effect=ValueError("bad value"),
        ):
            resp = client.put("/api/v1/config", json={"model": "bad"})
            assert resp.status_code == 400
            assert "invalid_config" in resp.json()["detail"]["error"]

    def test_get_service_status_no_api_key(self, client):
        with patch("subtitle_translator.api.routes.get_settings") as mock:
            settings = MagicMock()
            settings.openrouter_api_key = ""
            settings.openrouter_default_model = "test"
            mock.return_value = settings
            with patch("subtitle_translator.api.routes.get_translator") as mock_translator:
                translator = AsyncMock()
                mock_translator.return_value = translator
                resp = client.get("/api/v1/status")
                assert resp.status_code == 200
                assert resp.json()["healthy"] is True
