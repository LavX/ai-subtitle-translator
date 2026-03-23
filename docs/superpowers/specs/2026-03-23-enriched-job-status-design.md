# Enriched Job Status Endpoint

**Date:** 2026-03-23
**Status:** Approved

## Summary

Enrich the job status endpoint (`GET /api/v1/jobs/{job_id}`) to show all available information during and after processing: input metadata, processing metrics, cost, and user-provided labels.

## New Request Fields

Add to both `TranslateContentRequest` and `TranslateFileRequest`:

- `fileName` (`Optional[str]`) — original file name, purely informational
- `jobName` (`Optional[str]`) — user-provided label for the job

## Enriched Job Model

### Input metadata (set at submission time)

| Field | Type | Source |
|-------|------|--------|
| `jobName` | `Optional[str]` | User-provided |
| `fileName` | `Optional[str]` | User-provided |
| `sourceLanguage` | `Optional[str]` | From request |
| `targetLanguage` | `Optional[str]` | From request |
| `title` | `Optional[str]` | From request |
| `mediaType` | `Optional[str]` | From request |
| `model` | `Optional[str]` | From request or default |
| `totalLines` | `Optional[int]` | Computed from request lines/SRT content |

### Processing metrics (updated incrementally)

| Field | Type | Source |
|-------|------|--------|
| `totalBatches` | `Optional[int]` | Set when processing starts |
| `completedBatches` | `Optional[int]` | Updated per batch |
| `completedLines` | `Optional[int]` | Updated per batch |
| `tokensUsed` | `Optional[int]` | Accumulated from batch results |
| `totalCost` | `Optional[float]` | Accumulated from OpenRouter `usage.cost` |
| `elapsedSeconds` | `Optional[float]` | Computed: `(completedAt or now) - startedAt` |

## Data Flow

### At submission (routes.py -> job_manager.py)
1. Extract `fileName`, `jobName`, languages, title, mediaType, line count, model from request
2. Store as fields on `Job` via new `metadata` dict parameter in `submit_job()`

### During processing (worker.py -> job_manager.py)
1. Extend `update_progress()` to accept `tokens_used`, `cost`, `completed_lines`, `completed_batches`, `total_batches`
2. Worker progress_callback passes these from `BatchProgress`
3. Cost comes from `BatchResult.cost` -> `BatchProgress.total_cost`

### Cost capture (openrouter.py -> base.py)
1. Add `cost: Optional[float]` to `TranslationResult`
2. In `_process_response`, read `data.get("usage", {}).get("cost")`
3. Add `cost: float = 0.0` to `BatchResult`
4. Accumulate in `BatchProgress.total_cost`

### On read (routes.py)
1. `elapsedSeconds` computed: `(completedAt or now) - startedAt`
2. All fields returned in `JobStatusResponse`

## Files to modify

1. `src/subtitle_translator/api/models.py` — add request fields + response fields
2. `src/subtitle_translator/queue/job_manager.py` — enrich `Job` model + `update_progress()`
3. `src/subtitle_translator/queue/worker.py` — pass metrics in progress_callback
4. `src/subtitle_translator/providers/base.py` — add `cost` to `TranslationResult`
5. `src/subtitle_translator/providers/openrouter.py` — capture `usage.cost`
6. `src/subtitle_translator/core/batch_processor.py` — add `cost` to `BatchResult`/`BatchProgress`, propagate
7. `src/subtitle_translator/api/routes.py` — extract metadata at submission, compute `elapsedSeconds` on read
