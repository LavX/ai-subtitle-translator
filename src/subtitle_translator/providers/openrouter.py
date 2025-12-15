"""OpenRouter API implementation for subtitle translation."""

import json
import logging
from typing import Any, Optional

import httpx

from subtitle_translator.config import Settings, get_settings
from subtitle_translator.providers.base import (
    AuthenticationError,
    InvalidResponseError,
    RateLimitError,
    TranslationBatch,
    TranslationProvider,
    TranslationProviderError,
    TranslationResult,
)

logger = logging.getLogger(__name__)

# Recommended models for subtitle translation
RECOMMENDED_MODELS = [
    {
        "id": "google/gemini-2.5-flash-preview-09-2025",
        "name": "Gemini 2.5 Flash Preview",
        "description": "Fast and efficient for subtitle translation with excellent quality",
        "context_length": 1048576,
    },
    {
        "id": "google/gemini-2.5-flash-lite-preview-09-2025",
        "name": "Gemini 2.5 Flash Lite Preview",
        "description": "Lightweight version, very fast and cost-effective",
        "context_length": 1048576,
    },
    {
        "id": "anthropic/claude-sonnet-4.5",
        "name": "Claude Sonnet 4.5",
        "description": "Excellent quality translations with nuanced understanding",
        "context_length": 200000,
    },
    {
        "id": "anthropic/claude-haiku-4.5",
        "name": "Claude Haiku 4.5",
        "description": "Fast and affordable with good quality",
        "context_length": 200000,
    },
    {
        "id": "openai/gpt-5-nano",
        "name": "GPT-5 Nano",
        "description": "Compact and efficient OpenAI model",
        "context_length": 128000,
    },
    {
        "id": "openai/gpt-oss-120b",
        "name": "GPT OSS 120B",
        "description": "Large open-source style model from OpenAI",
        "context_length": 128000,
    },
    {
        "id": "x-ai/grok-4.1-fast",
        "name": "Grok 4.1 Fast",
        "description": "Fast xAI model with good translation capabilities",
        "context_length": 131072,
    },
    {
        "id": "meta-llama/llama-4-maverick",
        "name": "Llama 4 Maverick",
        "description": "Meta's latest Llama model, excellent multilingual support",
        "context_length": 128000,
    },
    {
        "id": "moonshotai/kimi-k2-0905",
        "name": "Kimi K2",
        "description": "Moonshot AI model with strong translation capabilities",
        "context_length": 128000,
    },
    {
        "id": "minimax/minimax-m2",
        "name": "MiniMax M2",
        "description": "Efficient model from MiniMax",
        "context_length": 128000,
    },
]


class OpenRouterProvider(TranslationProvider):
    """Translation provider using OpenRouter API."""

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize OpenRouter provider.

        Args:
            settings: Optional settings instance. Uses global settings if not provided.
        """
        self.settings = settings or get_settings()
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def provider_name(self) -> str:
        """Return the name of this provider."""
        return "openrouter"

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.settings.openrouter_api_base,
                headers=self.settings.openrouter_headers,
                timeout=httpx.Timeout(self.settings.request_timeout),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if OpenRouter is accessible and API key is valid."""
        if not self.settings.openrouter_api_key:
            return False

        try:
            response = await self.client.get("/models", timeout=10.0)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"OpenRouter health check failed: {e}")
            return False

    async def get_available_models(self) -> list[dict]:
        """
        Get list of recommended models for subtitle translation.

        Returns:
            List of model information dictionaries
        """
        default_model = self.settings.openrouter_default_model
        models = []
        
        for model in RECOMMENDED_MODELS:
            model_info = model.copy()
            model_info["is_default"] = model["id"] == default_model
            models.append(model_info)

        return models

    async def translate_batch(
        self,
        batch: TranslationBatch,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> TranslationResult:
        """
        Translate a batch of subtitle lines using OpenRouter.

        Args:
            batch: The batch of subtitle lines to translate
            model: Optional model override
            temperature: Optional temperature override

        Returns:
            TranslationResult containing translated lines and metadata

        Raises:
            TranslationProviderError: On translation failure
        """
        model_to_use = model or self.settings.openrouter_default_model
        temp_to_use = temperature if temperature is not None else self.settings.openrouter_temperature

        # Build messages
        system_prompt = self.build_system_prompt(
            batch.target_language,
            batch.context_title,
            batch.context_media_type,
        )
        user_content = self.format_input_for_translation(batch.lines)

        # Build request payload
        payload = {
            "model": model_to_use,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temp_to_use,
            "max_tokens": self.settings.openrouter_max_tokens,
            "response_format": {"type": "json_object"},
        }

        logger.debug(f"Sending translation request for {len(batch.lines)} lines using {model_to_use}")

        try:
            response = await self.client.post("/chat/completions", json=payload)
            return await self._process_response(response, model_to_use)
        except httpx.TimeoutException as e:
            raise TranslationProviderError(
                f"Request timed out after {self.settings.request_timeout}s",
                provider=self.provider_name,
                retryable=True,
            ) from e
        except httpx.RequestError as e:
            raise TranslationProviderError(
                f"Network error: {str(e)}",
                provider=self.provider_name,
                retryable=True,
            ) from e

    async def _process_response(
        self, response: httpx.Response, model_used: str
    ) -> TranslationResult:
        """
        Process the OpenRouter API response.

        Args:
            response: HTTP response from OpenRouter
            model_used: Model that was used for translation

        Returns:
            TranslationResult with parsed translations

        Raises:
            Various TranslationProviderError subclasses on failure
        """
        status_code = response.status_code

        # Handle error status codes
        if status_code == 401:
            raise AuthenticationError(
                "Invalid OpenRouter API key",
                provider=self.provider_name,
            )
        elif status_code == 429:
            retry_after = response.headers.get("retry-after")
            retry_seconds = float(retry_after) if retry_after else None
            raise RateLimitError(
                "OpenRouter rate limit exceeded",
                provider=self.provider_name,
                retry_after=retry_seconds,
            )
        elif status_code >= 500:
            raise TranslationProviderError(
                f"OpenRouter server error: {status_code}",
                provider=self.provider_name,
                retryable=True,
                status_code=status_code,
            )
        elif status_code >= 400:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", response.text)
            except Exception:
                error_msg = response.text
            raise TranslationProviderError(
                f"OpenRouter API error: {error_msg}",
                provider=self.provider_name,
                retryable=False,
                status_code=status_code,
            )

        # Parse successful response
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            raise InvalidResponseError(
                f"Invalid JSON response: {str(e)}",
                provider=self.provider_name,
                raw_response=response.text[:1000],
            ) from e

        # Extract usage information
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

        # Extract translated content
        choices = data.get("choices", [])
        if not choices:
            raise InvalidResponseError(
                "No choices in response",
                provider=self.provider_name,
                raw_response=str(data)[:1000],
            )

        content = choices[0].get("message", {}).get("content", "")
        if not content:
            raise InvalidResponseError(
                "Empty content in response",
                provider=self.provider_name,
                raw_response=str(data)[:1000],
            )

        # Parse the JSON array from content
        translations = self._parse_translations(content)

        return TranslationResult(
            translations=translations,
            model_used=model_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw_response=data,
        )

    def _parse_translations(self, content: str) -> list[dict[str, str]]:
        """
        Parse translation JSON from LLM response.

        Args:
            content: Raw content string from LLM response

        Returns:
            List of translation dictionaries

        Raises:
            InvalidResponseError: If JSON parsing fails
        """
        try:
            # Try to parse as is
            parsed = json.loads(content)
            
            # Handle case where response is wrapped in an object
            if isinstance(parsed, dict):
                # Look for common wrapper keys
                for key in ["translations", "results", "data", "lines", "subtitles"]:
                    if key in parsed and isinstance(parsed[key], list):
                        parsed = parsed[key]
                        break
                else:
                    # If no wrapper found but it's a single translation, wrap it
                    if "index" in parsed and "content" in parsed:
                        parsed = [parsed]
                    else:
                        raise InvalidResponseError(
                            f"Unexpected response structure: {list(parsed.keys())}",
                            provider=self.provider_name,
                            raw_response=content[:1000],
                        )

            if not isinstance(parsed, list):
                raise InvalidResponseError(
                    f"Expected list, got {type(parsed).__name__}",
                    provider=self.provider_name,
                    raw_response=content[:1000],
                )

            # Validate and normalize each translation entry
            translations = []
            for item in parsed:
                if not isinstance(item, dict):
                    logger.warning(f"Skipping non-dict item in translations: {item}")
                    continue
                    
                # Handle various key names
                index = str(item.get("index", item.get("idx", item.get("position", ""))))
                text = item.get("content", item.get("text", item.get("translation", "")))
                
                if index and text is not None:
                    translations.append({"index": index, "content": str(text)})

            return translations

        except json.JSONDecodeError as e:
            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
            if json_match:
                return self._parse_translations(json_match.group(1))
            
            # Try to find JSON array directly
            array_match = re.search(r"\[[\s\S]*\]", content)
            if array_match:
                try:
                    return self._parse_translations(array_match.group(0))
                except Exception:
                    pass

            raise InvalidResponseError(
                f"Failed to parse JSON: {str(e)}",
                provider=self.provider_name,
                raw_response=content[:1000],
            ) from e


async def get_openrouter_provider() -> OpenRouterProvider:
    """Factory function to get an OpenRouter provider instance."""
    return OpenRouterProvider()