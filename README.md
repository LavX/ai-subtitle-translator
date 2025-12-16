# 🎬 AI Subtitle Translator

> LLM-powered subtitle translation microservice by **LavX**

A standalone microservice for translating subtitles using AI/LLM via OpenRouter API. Originally created for [LavX's Bazarr fork](https://github.com/LavX/bazarr) but designed as a flexible API that can be integrated with any subtitle management system.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

## ✨ Features

- 🤖 **AI-Powered Translation**: Leverage state-of-the-art LLMs for high-quality subtitle translations
- 🌍 **Multi-language Support**: Translate between any language pairs supported by modern LLMs
- 🚀 **OpenRouter Integration**: Access multiple AI providers (Google Gemini, Claude, GPT, Llama, etc.) through a single API
- 📁 **SRT File Support**: Full SRT subtitle file parsing and generation
- 🔄 **Smart Batch Processing**: Efficient batch processing for large subtitle files with retry logic
- ↔️ **RTL Language Support**: Automatic directional markers for right-to-left languages (Arabic, Hebrew, Persian, etc.)
- 🔌 **Universal API**: RESTful API that integrates with any application - Bazarr, custom tools, or scripts
- 🐳 **Docker Ready**: Easy deployment with Docker and Docker Compose

## 🚀 Quick Start

### Using Docker (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/LavX/ai-subtitle-translator.git
cd ai-subtitle-translator
```

2. Create a `.env` file from the example:
```bash
cp .env.example .env
```

3. Add your OpenRouter API key to `.env`:
```bash
OPENROUTER_API_KEY=your_api_key_here
```

4. Start the service:
```bash
docker-compose up -d
```

The service will be available at `http://localhost:8765`

### Manual Installation

1. Install Python 3.11+ and create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variables:
```bash
export OPENROUTER_API_KEY=your_api_key_here
```

4. Run the service:
```bash
cd src && uvicorn subtitle_translator.main:app --host 0.0.0.0 --port 8765
```

## 📖 API Documentation

Once running, access the interactive API documentation at:
- **Swagger UI**: `http://localhost:8765/docs`
- **ReDoc**: `http://localhost:8765/redoc`

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/v1/models` | List available AI models |
| `POST` | `/api/v1/translate/content` | Translate subtitle lines |
| `POST` | `/api/v1/translate/file` | Translate entire SRT file |

### Translate Subtitle Content

```http
POST /api/v1/translate/content
Content-Type: application/json

{
  "sourceLanguage": "en",
  "targetLanguage": "es",
  "title": "Breaking Bad",
  "mediaType": "Episode",
  "lines": [
    {"position": 1, "line": "Hello, world!"},
    {"position": 2, "line": "How are you?"}
  ]
}
```

**Response:**
```json
{
  "lines": [
    {"position": 1, "line": "¡Hola, mundo!"},
    {"position": 2, "line": "¿Cómo estás?"}
  ],
  "model_used": "google/gemini-2.5-flash-preview-09-2025",
  "tokens_used": 150
}
```

### Translate SRT File

```http
POST /api/v1/translate/file
Content-Type: application/json

{
  "content": "1\n00:00:01,000 --> 00:00:04,000\nHello world\n\n2\n00:00:05,000 --> 00:00:08,000\nHow are you?\n",
  "sourceLanguage": "en",
  "targetLanguage": "es"
}
```

**Response:**
```json
{
  "content": "1\n00:00:01,000 --> 00:00:04,000\nHola mundo\n\n2\n00:00:05,000 --> 00:00:08,000\n¿Cómo estás?\n",
  "model_used": "google/gemini-2.5-flash-preview-09-2025",
  "tokens_used": 200,
  "subtitle_count": 2
}
```

## ⚙️ Configuration

All configuration is done via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key **(required)** | - |
| `OPENROUTER_DEFAULT_MODEL` | Default AI model for translation | `amazon/nova-2-lite-v1:free` |
| `OPENROUTER_TEMPERATURE` | Temperature for AI responses (0.0-2.0) | `0.3` |
| `OPENROUTER_MAX_TOKENS` | Maximum tokens per response | `8000` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8765` |
| `BATCH_SIZE` | Lines per translation batch | `100` |
| `MAX_RETRIES` | Maximum retry attempts on failure | `3` |
| `RETRY_DELAY` | Initial delay between retries (seconds) | `1.0` |
| `REQUEST_TIMEOUT` | Request timeout (seconds) | `120.0` |

## 🤖 Supported AI Models

The service supports any model available on [OpenRouter](https://openrouter.ai/models). Models have been extensively tested through a **Battle Royale** elimination process (5→10→20→30→40→50 lines at 80% translation threshold).

### 🏆 Battle Royale Champions (Survived ALL 6 Rounds)

These models consistently delivered high-quality translations across all test rounds:

#### Speed Champions ⚡
| Model | Avg Speed | Success Rate | Notes |
|-------|-----------|--------------|-------|
| `meta-llama/llama-4-maverick` | 3s | 92% | 🥇 Fastest overall (1-4s) |
| `google/gemini-2.5-flash-lite-preview-09-2025` | 3.2s | 90% | 🥈 Very fast, reasoning support |
| `moonshotai/kimi-k2-0905:exacto` | 4s | 92% | 🥉 Balanced speed & quality |

#### Quality Champions ⭐
| Model | Avg Speed | Success Rate | Notes |
|-------|-----------|--------------|-------|
| `google/gemini-2.5-flash-preview-09-2025` | 8.5s | 92% | Default model, reasoning support |
| `anthropic/claude-haiku-4.5` | 13s | 93% | Highest quality, premium |
| `anthropic/claude-sonnet-4.5` | 18s | 92% | Premium, nuanced translations |

### 🆓 Excellent FREE Models (Zero Cost!)

These free models survived ALL Battle Royale rounds:

| Model | Avg Speed | Success Rate | Notes |
|-------|-----------|--------------|-------|
| `amazon/nova-2-lite-v1:free` | 17s | 95% | 🏆 Best free model! |
| `nex-agi/deepseek-v3.1-nex-n1:free` | 35s | 90% | Reliable backup, slower |

### ⚠️ Models to Avoid

These models failed Battle Royale Round 1 (0% translation or timeout):

| Model | Reason |
|-------|--------|
| `tngtech/deepseek-r1t-chimera:free` | 0% translation output |
| `cognitivecomputations/dolphin-mistral-24b-venice-edition:free` | 0% translation output |
| `nvidia/nemotron-3-nano-30b-a3b:free` | 0% translation output |
| `allenai/olmo-3-32b-think:free` | Timeout |
| `openai/gpt-oss-120b:exacto` | Timeout |
| `x-ai/grok-4.1-fast` | Poor translation quality |

### Model Selection Guide

```
Need SPEED?      → meta-llama/llama-4-maverick
Need QUALITY?    → anthropic/claude-haiku-4.5
Need FREE?       → amazon/nova-2-lite-v1:free (default)
Balanced?        → google/gemini-2.5-flash-preview-09-2025
```

## 🔗 Integration

### Bazarr Integration

This service was designed for seamless integration with [LavX's Bazarr fork](https://github.com/LavX/bazarr):

1. Deploy the AI Subtitle Translator service
2. Configure Bazarr to use this service as a translation provider
3. Enjoy AI-powered subtitle translations!

### Custom Integration

The REST API makes it easy to integrate with any application:

```python
import httpx

async def translate_subtitles(lines, source_lang, target_lang):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8765/api/v1/translate/content",
            json={
                "sourceLanguage": source_lang,
                "targetLanguage": target_lang,
                "lines": [{"position": i, "line": line} for i, line in enumerate(lines)]
            }
        )
        return response.json()
```

## 🛠️ Development

### Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
# Format code
black src/ tests/
ruff check src/ tests/ --fix

# Type checking
mypy src/
```

## 📁 Project Structure

```
ai-subtitle-translator/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── README.md
├── .env.example
├── src/
│   └── subtitle_translator/
│       ├── main.py                 # FastAPI application
│       ├── config.py               # Configuration management
│       ├── api/
│       │   ├── routes.py           # API endpoints
│       │   └── models.py           # Pydantic models
│       ├── core/
│       │   ├── translator.py       # Translation orchestration
│       │   ├── srt_parser.py       # SRT file handling
│       │   └── batch_processor.py  # Batch processing
│       └── providers/
│           ├── base.py             # Abstract provider
│           └── openrouter.py       # OpenRouter implementation
└── tests/
    ├── test_api.py
    └── test_translator.py
```

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Credits

Created by **[LavX](https://github.com/LavX)**

- Originally developed for [Bazarr](https://github.com/LavX/bazarr)
- Powered by [OpenRouter](https://openrouter.ai/) for AI model access
- Built with [FastAPI](https://fastapi.tiangolo.com/)

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest features
- Submit pull requests

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/LavX">LavX</a>
</p>