# AI Subtitle Translator

LLM-powered subtitle translation API using [OpenRouter](https://openrouter.ai/). Translate SRT files and subtitle content through any OpenRouter-supported model (Gemini, Claude, Llama, GPT, Mistral, and more). Built as a standalone microservice for [LavX's Bazarr fork](https://github.com/LavX/bazarr), but works with any HTTP client.

**Keywords:** subtitle translation API, SRT translator, AI subtitle translation, LLM translation service, OpenRouter translation, automated subtitle localization, Bazarr AI translation, multilingual subtitle converter

[![CI](https://github.com/LavX/ai-subtitle-translator/actions/workflows/ci.yml/badge.svg)](https://github.com/LavX/ai-subtitle-translator/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

## Features

- Translate SRT subtitle files or raw subtitle lines via REST API
- Routes through OpenRouter to 300+ LLMs (Gemini, Claude, Llama, GPT, Mercury, Mistral, etc.)
- Automatic batch sizing per model. Adapts to model context windows and retries with smaller batches on failure
- RTL language support (Arabic, Hebrew, Persian, Urdu, etc.)
- Async job queue with real-time progress tracking, cost reporting, and per-batch status updates
- Job persistence via SQLite. Jobs survive restarts, automatically recovered on startup
- AES-256-GCM API key encryption between any client and this service (optional, enabled by default)
- Dynamic reasoning detection. Fetches model capabilities from OpenRouter API, no hardcoded lists
- Rate limit handling with exponential backoff, serialized retries, and staggered parallel requests
- Per-request config overrides: model, temperature, API key, reasoning, parallel batch count

## Quick start

### Docker

```bash
git clone https://github.com/LavX/ai-subtitle-translator.git
cd ai-subtitle-translator

# Set your OpenRouter API key
echo "OPENROUTER_API_KEY=sk-or-..." > .env

docker compose up -d
```

Service runs at `http://localhost:8765`. Interactive docs at `/docs`.

### Manual

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...
cd src && uvicorn subtitle_translator.main:app --host 0.0.0.0 --port 8765
```

## API endpoints

Full interactive docs at `/docs` (Swagger) or `/redoc` when running.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/v1/models` | List available translation models |
| `GET` | `/api/v1/status` | Service status and queue stats |
| `GET` | `/api/v1/config` | Current configuration |
| `PUT` | `/api/v1/config` | Update config at runtime |
| `POST` | `/api/v1/translate/content` | Translate subtitle lines (synchronous) |
| `POST` | `/api/v1/translate/file` | Translate SRT file (synchronous) |
| `POST` | `/api/v1/jobs/translate/content` | Submit async translation job |
| `POST` | `/api/v1/jobs/translate/file` | Submit async SRT translation job |
| `GET` | `/api/v1/jobs` | List all jobs (filter by status) |
| `GET` | `/api/v1/jobs/{id}` | Job status, progress, metrics, and result |
| `DELETE` | `/api/v1/jobs/{id}` | Cancel or delete a job |

### Translate subtitle content

```bash
curl -X POST http://localhost:8765/api/v1/translate/content \
  -H "Content-Type: application/json" \
  -d '{
    "sourceLanguage": "en",
    "targetLanguage": "hu",
    "title": "Breaking Bad",
    "lines": [
      {"position": 1, "line": "Say my name."},
      {"position": 2, "line": "You are goddamn right."}
    ]
  }'
```

### Submit async translation job

```bash
curl -X POST http://localhost:8765/api/v1/jobs/translate/content \
  -H "Content-Type: application/json" \
  -d '{
    "sourceLanguage": "en",
    "targetLanguage": "hu",
    "title": "Breaking Bad S05E07",
    "mediaType": "Episode",
    "fileName": "breaking.bad.s05e07.srt",
    "jobName": "bb-s05e07-hungarian",
    "lines": [
      {"position": 1, "line": "Say my name."}
    ],
    "config": {
      "model": "anthropic/claude-haiku-4.5",
      "temperature": 0.3,
      "reasoning": {"effort": "high"},
      "parallelBatches": 4
    }
  }'
```

### Job status response

Poll `GET /api/v1/jobs/{id}` for live progress and metrics:

```json
{
  "jobId": "3f1318c0-...",
  "status": "processing",
  "progress": 66,
  "message": "Translated 400/600 lines (4/6 batches)",
  "jobName": "bb-s05e07-hungarian",
  "fileName": "breaking.bad.s05e07.srt",
  "sourceLanguage": "en",
  "targetLanguage": "hu",
  "title": "Breaking Bad S05E07",
  "mediaType": "Episode",
  "model": "anthropic/claude-haiku-4.5",
  "totalLines": 600,
  "totalBatches": 6,
  "completedBatches": 4,
  "completedLines": 400,
  "tokensUsed": 25000,
  "totalCost": 0.0084,
  "elapsedSeconds": 12.5
}
```

Job statuses: `queued` | `processing` | `completed` | `partial` | `failed` | `cancelled`

### Per-request config override

Every translate endpoint accepts an optional `config` block to override defaults:

```json
{
  "config": {
    "apiKey": "sk-or-different-key",
    "model": "anthropic/claude-haiku-4.5",
    "temperature": 0.5,
    "parallelBatches": 2,
    "reasoning": {"effort": "high"}
  }
}
```

Reasoning effort levels: `xhigh`, `high`, `medium`, `low`, `minimal`, `none` (disables reasoning).

## Configuration

Set via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | *(required)* | Your OpenRouter API key |
| `OPENROUTER_DEFAULT_MODEL` | `amazon/nova-2-lite-v1:free` | Default translation model |
| `OPENROUTER_TEMPERATURE` | `0.3` | Sampling temperature |
| `BATCH_SIZE` | `100` | Max subtitle lines per batch (auto-adjusted per model) |
| `PARALLEL_BATCHES_PER_JOB` | `4` | Concurrent batches per translation job |
| `MAX_RETRIES` | `3` | Retry attempts on failure (rate limits get 3 extra) |
| `REQUEST_TIMEOUT` | `120.0` | HTTP request timeout in seconds |
| `LOG_LEVEL` | `INFO` | Log level (`DEBUG` for full request/response logging) |
| `CORS_ALLOWED_ORIGINS` | `*` | Comma-separated allowed CORS origins |
| `ADMIN_API_KEY` | *(empty)* | Required as `X-Admin-Key` header for PUT /config when set |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8765` | Server port |
| `ENCRYPTION_ENABLED` | `true` | Enable AES-256-GCM API key encryption |
| `ENCRYPTION_KEY` | *(auto-generated)* | 64-char hex AES-256 key. Overrides key file when set |
| `ENCRYPTION_KEY_FILE` | `/app/data/encryption.key` | Path to persistent encryption key file |
| `DB_PATH` | `/app/data/jobs.db` | SQLite database path for job persistence |
| `JOB_RETENTION_HOURS` | `24` | Hours to keep completed/failed jobs before cleanup |

## Job persistence

Jobs are stored in SQLite and survive container restarts. On startup, any queued or in-progress jobs from the previous session are automatically recovered and re-queued.

Mount a volume to `/app/data` to persist across container recreations:

```bash
docker run -d --name ai-subtitle-translator \
  -v /path/to/data:/app/data \
  -e OPENROUTER_API_KEY=sk-or-... \
  ghcr.io/lavx/ai-subtitle-translator:latest
```

The data directory contains:
- `jobs.db` - SQLite database with all job history
- `encryption.key` - auto-generated encryption key (chmod 600)

## API key encryption

API keys sent between Bazarr and the translator can be encrypted in transit using AES-256-GCM with a pre-shared key. This prevents API keys from being visible in plaintext on the network, even over HTTP.

### How it works

1. On first startup, the translator generates an encryption key and prints it to the logs:

```
======================================================================
NEW ENCRYPTION KEY GENERATED
Key: e2c65e79252379c1da0874f9d0928c6194fbc5e5460e3ab4047860560a2aa597
Copy this key to your Bazarr AI Subtitle Translator settings.
Saved to: /app/data/encryption.key
This key will only be shown in full once.
======================================================================
```

2. Copy this key to Bazarr's AI Subtitle Translator provider settings
3. Bazarr encrypts the OpenRouter API key before sending it in requests
4. The translator decrypts it on receipt

Encrypted API keys use the format `enc:base64data`. Plaintext keys are still accepted for backward compatibility.

### Regenerate key

```bash
# Via CLI (inside container)
docker exec ai-subtitle-translator python -m subtitle_translator.cli regenerate-key

# Or set via environment variable (overrides key file)
docker run -e ENCRYPTION_KEY=your64charhexkey... ...
```

### Disable encryption

```bash
docker run -e ENCRYPTION_ENABLED=false ...
```

## Adaptive batch sizing

Different LLMs handle different batch sizes. Small-context models choke on 100 lines, while large-context models handle them fine. The translator adjusts automatically:

1. **Known limits** - small-context models get smaller default batches
2. **Context-length heuristic** - estimates safe batch size from the model's context window
3. **Adaptive retry** - if a batch fails, halves the size and retries. Remembers the safe size for future requests to the same model (in-memory, resets on restart)

You can use any OpenRouter model and the service will find the right batch size.

## Rate limit handling

When OpenRouter returns 429 Too Many Requests:
- Retries up to 6 times with exponential backoff (5s, 10s, 20s, 30s cap)
- Serializes retries across parallel batches so they don't all hit the API at once
- Staggers initial parallel requests by 0.5s to spread the load
- Reports `partial` status if some batches completed before retries ran out

## Reasoning support

Model reasoning capabilities are detected automatically from the OpenRouter `/models` API. No hardcoded model lists to maintain. When reasoning is requested:
- Models supporting `effort` get `{"reasoning": {"effort": "high"}}`
- Models supporting `max_tokens` get `{"reasoning": {"max_tokens": N}}`
- `effort: "none"` disables reasoning entirely
- `response_format: json_object` is skipped when reasoning is active (prevents single-object response bugs with certain models)

## Tested models for subtitle translation

Models tested through elimination rounds (5/10/20/30/40/50 subtitle lines, 80% accuracy threshold for Hungarian).

| Model | Speed | Accuracy | Notes |
|-------|-------|----------|-------|
| `meta-llama/llama-4-maverick` | ~3s | 92% | Fastest |
| `anthropic/claude-haiku-4.5` | ~13s | 93% | Best accuracy |
| `amazon/nova-2-lite-v1:free` | ~17s | 95% | Best free model |
| `google/gemini-2.5-flash-preview-09-2025` | ~8.5s | 92% | Good all-rounder |
| `inception/mercury-2` | ~3s | ~90% | ~$0.10/episode, 580+ tokens/sec |

Full model list with metadata: `GET /api/v1/models`

## Development

```bash
pip install -e ".[dev]"
pytest                    # 234 tests
ruff check src/ tests/    # lint
ruff format src/ tests/   # format
```

## Project structure

```
src/subtitle_translator/
  main.py                 # FastAPI application
  config.py               # Settings from environment variables
  crypto.py               # AES-256-GCM encryption and key management
  cli.py                  # CLI commands (key regeneration)
  api/
    routes.py             # REST API endpoints
    models.py             # Pydantic request/response models
  core/
    translator.py         # Translation orchestration
    srt_parser.py         # SRT file parsing and composing
    batch_processor.py    # Parallel batch processing with adaptive retry
    batch_sizing.py       # Per-model batch size resolution
  providers/
    base.py               # Abstract translation provider interface
    openrouter.py         # OpenRouter API implementation
  queue/
    job_manager.py        # Async job queue with progress tracking
    job_store.py          # SQLite persistence layer
    worker.py             # Background job worker
```

## License

MIT. See [LICENSE](LICENSE).

## Links

- [LavX](https://lavx.hu) - Enterprise AI solutions
- [LavX's Bazarr fork](https://github.com/LavX/bazarr) - Automated subtitle management with AI translation
- [OpenRouter](https://openrouter.ai/) - Multi-model LLM routing API
