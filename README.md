# AI Subtitle Translator

Subtitle translation microservice using LLMs via [OpenRouter](https://openrouter.ai/). Ships as a standalone API — built for [LavX's Bazarr fork](https://github.com/LavX/bazarr) but works with anything that speaks HTTP.

[![CI](https://github.com/LavX/ai-subtitle-translator/actions/workflows/ci.yml/badge.svg)](https://github.com/LavX/ai-subtitle-translator/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

## What it does

- Translates SRT subtitle files or raw subtitle lines via REST API
- Routes through OpenRouter to any supported LLM (Gemini, Claude, Llama, etc.)
- Batches large files automatically — adapts batch size per model capability
- Handles RTL languages (Arabic, Hebrew, Persian, etc.)
- Async job queue for long-running translations
- Retries with exponential backoff on failures

## Quick start

### Docker

```bash
git clone https://github.com/LavX/ai-subtitle-translator.git
cd ai-subtitle-translator

# Set your OpenRouter API key
echo "OPENROUTER_API_KEY=sk-or-..." > .env

docker compose up -d
```

Service runs at `http://localhost:8765`. Docs at `/docs`.

### Manual

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...
cd src && uvicorn subtitle_translator.main:app --host 0.0.0.0 --port 8765
```

## API

Full docs at `/docs` (Swagger) or `/redoc` when running.

| Method | Endpoint | What it does |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/v1/models` | List available models |
| `GET` | `/api/v1/config` | Current configuration |
| `PUT` | `/api/v1/config` | Update config at runtime |
| `POST` | `/api/v1/translate/content` | Translate subtitle lines |
| `POST` | `/api/v1/translate/file` | Translate an SRT file |
| `POST` | `/api/v1/jobs/translate/content` | Async translation job |
| `POST` | `/api/v1/jobs/translate/file` | Async SRT translation job |
| `GET` | `/api/v1/jobs/{id}` | Job status / result |

### Translate content

```bash
curl -X POST http://localhost:8765/api/v1/translate/content \
  -H "Content-Type: application/json" \
  -d '{
    "sourceLanguage": "en",
    "targetLanguage": "hu",
    "title": "Breaking Bad",
    "lines": [
      {"position": 1, "line": "Say my name."},
      {"position": 2, "line": "You're goddamn right."}
    ]
  }'
```

### Translate SRT file

```bash
curl -X POST http://localhost:8765/api/v1/translate/file \
  -H "Content-Type: application/json" \
  -d '{
    "content": "1\n00:00:01,000 --> 00:00:04,000\nSay my name.\n\n2\n00:00:05,000 --> 00:00:08,000\nYou'\''re goddamn right.\n",
    "sourceLanguage": "en",
    "targetLanguage": "hu"
  }'
```

### Per-request config override

Any translate endpoint accepts an optional `config` block to override model, temperature, API key, etc. per request:

```json
{
  "sourceLanguage": "en",
  "targetLanguage": "hu",
  "lines": [...],
  "config": {
    "model": "anthropic/claude-haiku-4.5",
    "temperature": 0.5,
    "api_key": "sk-or-different-key"
  }
}
```

## Configuration

Set via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | *(required)* | OpenRouter API key |
| `OPENROUTER_DEFAULT_MODEL` | `amazon/nova-2-lite-v1:free` | Default LLM |
| `OPENROUTER_TEMPERATURE` | `0.3` | Generation temperature |
| `OPENROUTER_MAX_TOKENS` | `8000` | Max tokens per response |
| `BATCH_SIZE` | `100` | Max lines per batch (auto-adjusted per model) |
| `PARALLEL_BATCHES_PER_JOB` | `4` | Concurrent batches per job |
| `MAX_RETRIES` | `3` | Retry attempts on failure |
| `REQUEST_TIMEOUT` | `120.0` | Request timeout (seconds) |
| `CORS_ALLOWED_ORIGINS` | `*` | Comma-separated allowed origins |
| `ADMIN_API_KEY` | *(empty)* | If set, required as `X-Admin-Key` header for PUT /config |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8765` | Server port |

## Adaptive batch sizing

Not all models handle large batches well — some truncate output, return garbage, or timeout. The translator automatically adjusts batch sizes per model:

1. **Known limits** — small-context models get hardcoded smaller batches
2. **Context-length heuristic** — estimates safe batch size from the model's context window
3. **Adaptive retry** — if a batch fails, halves the size and retries. Remembers the safe size for future requests (in-memory, resets on restart)

This means you can throw any model at it and it'll figure out the right batch size.

## Tested models

Models were tested through a battle royale elimination (5/10/20/30/40/50 lines, 80% threshold for Hungarian translation).

**Fast:** `meta-llama/llama-4-maverick` (3s, 92%) — best speed
**Quality:** `anthropic/claude-haiku-4.5` (13s, 93%) — best accuracy
**Free:** `amazon/nova-2-lite-v1:free` (17s, 95%) — best free option
**Balanced:** `google/gemini-2.5-flash-preview-09-2025` (8.5s, 92%)

Full results available via `GET /api/v1/models`.

## Development

```bash
pip install -e ".[dev]"
pytest                    # run tests
ruff check src/ tests/    # lint
```

## Project structure

```
src/subtitle_translator/
  main.py                 # FastAPI app
  config.py               # Settings (env vars)
  api/
    routes.py             # API endpoints
    models.py             # Request/response models
  core/
    translator.py         # Translation orchestration
    srt_parser.py         # SRT parsing/composing
    batch_processor.py    # Batch processing + adaptive retry
    batch_sizing.py       # Per-model batch size resolution
  providers/
    base.py               # Abstract provider interface
    openrouter.py         # OpenRouter implementation
  queue/
    job_manager.py        # Async job queue
    worker.py             # Job worker
```

## License

MIT — see [LICENSE](LICENSE).

## Links

- [LavX](https://lavx.hu) — Enterprise AI solutions
- [LavX's Bazarr fork](https://github.com/LavX/bazarr) — Automated subtitle management with AI translation
- [OpenRouter](https://openrouter.ai/) — LLM routing API
