"""Deterministic UI test server using the real API, queue and file worker.

Run only for browser checks. All model responses are produced locally.
"""

import argparse
import asyncio
import os
from pathlib import Path

from subtitle_translator.providers.base import (
    TranslationProvider,
    TranslationProviderError,
    TranslationResult,
)

FIXTURE_KEY = "22" * 32
DEMO_KEYS = {"sk-or-v1-demo-not-a-real-key", "sk-or-v1-demo-second-key"}


async def validate_demo_key(api_key):
    from fastapi import HTTPException

    if api_key not in DEMO_KEYS:
        raise HTTPException(status_code=401, detail="OpenRouter key was not accepted.")


class FixtureProvider(TranslationProvider):
    @property
    def provider_name(self):
        return "fixture"

    async def health_check(self):
        return True

    async def close(self):
        pass

    async def get_available_models(self):
        return [{"id": "fixture/model", "name": "Local test model", "is_default": True}]

    def get_model_metadata(self, model_id):
        return {"max_batch_size": 1, "context_length": 8192}

    async def translate_batch(self, batch, model=None, temperature=None, config_override=None):
        if config_override is None or config_override.api_key not in DEMO_KEYS:
            raise TranslationProviderError("Fixture requires the submitted key", provider="fixture")
        content = " ".join(line["content"] for line in batch.lines)
        if "SLOW" in content:
            await asyncio.sleep(6)
        if "FAIL" in content:
            raise TranslationProviderError("Fixture translation failure", provider="fixture")
        return TranslationResult(
            translations=[
                {"index": line["index"], "content": "HU: " + line["content"]}
                for line in batch.lines
            ],
            model_used=model or "fixture/model",
            total_tokens=10,
            cost=0.001,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--reliability-provider", action="store_true")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument(
        "--no-encryption",
        action="store_true",
        help="Exercise legacy API isolation without encryption",
    )
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        UI_ENABLED="true",
        DEBUG="false",
        LOG_LEVEL="ERROR",
        OPENROUTER_API_KEY="fixture-not-a-real-key",
        OPENROUTER_DEFAULT_MODEL="fixture/model",
        BATCH_SIZE="10" if args.reliability_provider else "1",
        PARALLEL_BATCHES_PER_JOB="1",
        JOB_QUEUE_MAX_CONCURRENT="1",
        MAX_RETRIES="1" if args.reliability_provider else "0",
        RETRY_DELAY="4",
        ENCRYPTION_ENABLED="false" if args.no_encryption else "true",
        ENCRYPTION_KEY=FIXTURE_KEY,
        DB_PATH=str(args.data_dir / "jobs.db"),
    )
    import subtitle_translator.core.translator as translator_module
    import subtitle_translator.ui_api as ui_api
    from subtitle_translator.config import get_settings
    from subtitle_translator.main import create_app

    async def fixture_catalog():
        return [
            ui_api.ModelInfo(id="openai/gpt-5.6-luna", name="GPT-5.6 Luna", is_default=True),
            ui_api.ModelInfo(id="outside/curated-list", name="External model"),
        ]

    ui_api._validate_openrouter_key = validate_demo_key
    ui_api._validated_keys.clear()
    if args.reliability_provider:
        from ui_reliability_fixture import install_provider

        provider = install_provider(get_settings(), args.evidence)
    else:
        ui_api._fetch_model_catalog = fixture_catalog
        provider = FixtureProvider()
    translator_module._translator_instance = translator_module.SubtitleTranslator(
        provider=provider, settings=get_settings()
    )
    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="error")


if __name__ == "__main__":
    main()
