"""OpenRouter-key authentication and private jobs for the optional browser UI."""

import hashlib
import hmac
from collections import OrderedDict
from dataclasses import dataclass, field
from time import monotonic
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from subtitle_translator.api.models import (
    JobDeleteResponse,
    JobListResponse,
    JobStatusResponse,
    JobSubmitResponse,
    ModelInfo,
    ModelReasoningInfo,
    ModelsResponse,
    TranslateFileRequest,
    TranslationConfig,
)
from subtitle_translator.api.routes import _build_job_status_response
from subtitle_translator.queue.job_manager import Job, JobStatus, JobType, job_manager

_validated_keys: OrderedDict[str, float] = OrderedDict()
_CACHE_TTL_SECONDS = 300
_CACHE_MAX_KEYS = 128
_PRIVATE_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


def _error(status_code: int, error: str, message: str) -> HTTPException:
    headers = dict(_PRIVATE_HEADERS)
    if status_code == 401:
        headers["WWW-Authenticate"] = "Bearer"
    return HTTPException(status_code, detail={"error": error, "message": message}, headers=headers)


class PrivateApiRoute(APIRoute):
    """Keep private responses uncached and validation errors free of submitted values."""

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def private_handler(request: Request):
            try:
                response = await handler(request)
            except RequestValidationError:
                return JSONResponse(
                    status_code=422,
                    content={"detail": {"error": "invalid_request", "message": "Invalid request"}},
                    headers=_PRIVATE_HEADERS,
                )
            except HTTPException as exc:
                exc.headers = {**_PRIVATE_HEADERS, **(exc.headers or {})}
                raise
            response.headers.update(_PRIVATE_HEADERS)
            return response

        return private_handler


@dataclass(frozen=True)
class UiIdentity:
    api_key: str = field(repr=False)
    owner: str


async def _validate_openrouter_key(api_key: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/key",
                headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            )
    except httpx.HTTPError:
        raise _error(
            503, "provider_unavailable", "OpenRouter key validation is unavailable"
        ) from None
    if response.status_code in (401, 403):
        raise _error(401, "invalid_api_key", "OpenRouter rejected this API key")
    if response.status_code == 429:
        raise _error(429, "rate_limited", "OpenRouter key validation is rate limited")
    if response.status_code != 200:
        raise _error(503, "provider_unavailable", "OpenRouter key validation is unavailable")
    try:
        payload = response.json()
    except ValueError:
        payload = None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict) or not isinstance(data.get("is_free_tier"), bool):
        raise _error(
            503, "provider_unavailable", "OpenRouter returned an invalid validation response"
        )


async def _authenticate(
    authorization: Annotated[str | None, Header()] = None,
) -> UiIdentity:
    scheme, separator, api_key = (authorization or "").partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not api_key
        or len(api_key) > 500
        or any(ord(character) < 33 or ord(character) > 126 for character in api_key)
    ):
        raise _error(401, "invalid_api_key", "An OpenRouter API key is required")
    owner = hashlib.sha256(b"subtitle-translator-ui\0" + api_key.encode()).hexdigest()
    now = monotonic()
    for fingerprint, expires in list(_validated_keys.items()):
        if expires <= now:
            del _validated_keys[fingerprint]
    if owner not in _validated_keys:
        await _validate_openrouter_key(api_key)
        _validated_keys[owner] = monotonic() + _CACHE_TTL_SECONDS
        while len(_validated_keys) > _CACHE_MAX_KEYS:
            _validated_keys.popitem(last=False)
    _validated_keys.move_to_end(owner)
    return UiIdentity(api_key=api_key, owner=owner)


Identity = Annotated[UiIdentity, Depends(_authenticate)]
ui_api_router = APIRouter(prefix="/ui/api", route_class=PrivateApiRoute, include_in_schema=False)


def _belongs_to(job: Job, identity: UiIdentity) -> bool:
    owner = job.request_data.get("_ui_owner")
    return isinstance(owner, str) and hmac.compare_digest(owner.encode(), identity.owner.encode())


def _owned_job(job_id: str, identity: UiIdentity) -> Job:
    job = job_manager.get_job(job_id)
    if job is None or not _belongs_to(job, identity):
        raise _error(404, "job_not_found", "Job not found")
    return job


@ui_api_router.post("/connect")
async def connect(identity: Identity) -> dict:
    return {"connected": True, "ownerScope": identity.owner}


UI_DEFAULT_MODEL = "openai/gpt-5.6-luna:floor"
_catalog_cache: tuple[float, list[ModelInfo]] | None = None


async def _fetch_model_catalog() -> list[ModelInfo]:
    global _catalog_cache
    if _catalog_cache and _catalog_cache[0] > monotonic():
        return _catalog_cache[1]
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=False) as client:
            response = await client.get("https://openrouter.ai/api/v1/models")
            response.raise_for_status()
            data = response.json()["data"]
            if not isinstance(data, list) or not data:
                raise ValueError("Invalid catalog")
            models_by_id = {}
            for model in data:
                reasoning = model.get("reasoning")
                choices = None
                if isinstance(reasoning, dict):
                    efforts = reasoning.get("supported_efforts")
                    choices = ModelReasoningInfo(
                        mandatory=(
                            reasoning["mandatory"]
                            if isinstance(reasoning.get("mandatory"), bool)
                            else None
                        ),
                        supported_efforts=list(
                            dict.fromkeys(
                                effort
                                for effort in (efforts if isinstance(efforts, list) else [])
                                if isinstance(effort, str)
                                and effort in ("none", "minimal", "low", "medium", "high", "xhigh")
                            )
                        ),
                    )
                entry = ModelInfo(
                    id=model["id"],
                    name=model["name"],
                    context_length=model.get("context_length"),
                    pricing=model.get("pricing"),
                    is_default=model["id"] == "openai/gpt-5.6-luna",
                    reasoning=choices,
                )
                models_by_id.setdefault(entry.id, entry)
            entries = sorted(
                models_by_id.values(), key=lambda item: (not item.is_default, item.name.casefold())
            )
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        raise _error(
            503, "catalog_unavailable", "OpenRouter model catalog is unavailable"
        ) from None
    _catalog_cache = (monotonic() + _CACHE_TTL_SECONDS, entries)
    return entries


@ui_api_router.get("/models", response_model=ModelsResponse)
async def models(identity: Identity) -> ModelsResponse:
    return ModelsResponse(models=await _fetch_model_catalog(), default_model=UI_DEFAULT_MODEL)


@ui_api_router.post("/jobs/translate/file", response_model=JobSubmitResponse)
async def submit_file(request: TranslateFileRequest, identity: Identity) -> JobSubmitResponse:
    return await submit_gui_file(request, identity)


async def submit_gui_file(
    request: TranslateFileRequest, identity: UiIdentity, submission_id: str | None = None
) -> JobSubmitResponse:
    if request.config is not None and request.config.api_key is not None:
        raise _error(
            422, "invalid_request", "Supply the OpenRouter key only in the Authorization header"
        )
    if not request.content.strip():
        raise _error(400, "invalid_request", "SRT content is required")
    if request.config is None:
        request.config = TranslationConfig(model=request.model or UI_DEFAULT_MODEL)
    elif not request.config.model:
        request.config.model = request.model or UI_DEFAULT_MODEL
    request_data = request.model_dump(exclude={"config"})
    if request.config is not None:
        request_data["config"] = request.config.model_dump(exclude={"api_key"}, exclude_none=True)
    request_data["_ui_owner"] = identity.owner
    if submission_id:
        for existing in job_manager.list_jobs(limit=len(job_manager.jobs)):
            if (
                _belongs_to(existing, identity)
                and existing.request_data.get("_ui_submission") == submission_id
            ):
                return JobSubmitResponse(
                    jobId=existing.id,
                    status=existing.status.value,
                    position=job_manager.get_queue_position(existing.id),
                )
        request_data["_ui_submission"] = submission_id
    resolved_model = (
        (request.config.model if request.config else None) or request.model or UI_DEFAULT_MODEL
    )
    try:
        job_id = await job_manager.submit_job(
            request_data=request_data,
            job_type=JobType.TRANSLATE_FILE,
            api_key_override=identity.api_key,
            metadata={
                "job_name": request.jobName,
                "file_name": request.fileName,
                "source_language": request.sourceLanguage,
                "target_language": request.targetLanguage,
                "title": request.title,
                "media_type": request.mediaType,
                "model": resolved_model,
                "total_lines": None,
            },
        )
    except RuntimeError:
        raise _error(429, "queue_full", "The translation queue is full") from None
    return JobSubmitResponse(jobId=job_id, position=job_manager.get_queue_position(job_id))


@ui_api_router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    identity: Identity, limit: int = Query(default=100, ge=1, le=1000)
) -> JobListResponse:
    jobs = [
        job
        for job in job_manager.list_jobs(limit=len(job_manager.jobs))
        if _belongs_to(job, identity)
    ]
    return JobListResponse(
        jobs=[_build_job_status_response(job) for job in jobs[:limit]],
        total=len(jobs),
        processing=sum(job.status == JobStatus.PROCESSING for job in jobs),
        queued=sum(job.status == JobStatus.QUEUED for job in jobs),
    )


@ui_api_router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str, identity: Identity) -> JobStatusResponse:
    return _build_job_status_response(_owned_job(job_id, identity))


@ui_api_router.delete("/jobs/{job_id}", response_model=JobDeleteResponse)
async def cancel_job(job_id: str, identity: Identity) -> JobDeleteResponse:
    job = _owned_job(job_id, identity)
    if job.status == JobStatus.QUEUED:
        job_manager.cancel_job(job_id)
        message = "Job cancelled successfully"
    elif job.status == JobStatus.PROCESSING:
        if job_manager.cancel_job(job_id):
            # The worker records the final status once the handler has stopped.
            status = "cancelled" if job.status == JobStatus.CANCELLED else "cancelling"
            return JobDeleteResponse(jobId=job_id, status=status, message="Cancellation requested")
        message = "Cannot cancel job that is currently processing"
    else:
        message = "Job is no longer queued; kept unchanged"
    return JobDeleteResponse(jobId=job_id, status=job.status.value, message=message)


async def forget_job(job_id: str, identity: Identity) -> dict:
    """Delete a finished job and its stored result. Active jobs are refused."""
    job = _owned_job(job_id, identity)
    if job.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
        raise HTTPException(
            status_code=409,
            detail={"error": "active", "message": "Cancel the job before forgetting it"},
        )
    job_manager.delete_job(job_id)
    return {"jobId": job_id, "deleted": True}
