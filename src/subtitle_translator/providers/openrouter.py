"""OpenRouter API implementation for subtitle translation."""

import json
import logging
from typing import Any, Optional, TYPE_CHECKING

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

if TYPE_CHECKING:
    from subtitle_translator.api.models import TranslationConfig

logger = logging.getLogger(__name__)

# Debug logger for detailed request/response logging
debug_logger = logging.getLogger(f"{__name__}.debug")

# Recommended models for subtitle translation - Updated based on testing
# Models marked as working after comprehensive testing (Dec 2025)

# === BATTLE ROYALE TESTED MODELS - Dec 2025 ===
# Champions that survived ALL 6 ROUNDS (5→10→20→30→40→50 lines at 80% threshold)

# EXCELLENT: TOP TIER - All champions from Battle Royale
EXCELLENT_MODELS = [
    # 🏆 SPEED CHAMPIONS
    {
        "id": "meta-llama/llama-4-maverick",
        "name": "Llama 4 Maverick",
        "description": "🏆 SPEED CHAMPION - Fastest in Battle Royale (1-4s), 80-92% Hungarian, survived ALL rounds",
        "context_length": 1048576,
        "supports_reasoning": False,
        "reasoning_type": None,
        "recommended_for": ["hungarian", "speed", "battle_royale_champion"],
        "success_rate": 92,
        "avg_speed_seconds": 3.0,
    },
    {
        "id": "google/gemini-2.5-flash-lite-preview-09-2025",
        "name": "Gemini 2.5 Flash Lite",
        "description": "🥈 SPEED RUNNER-UP - Very fast (1-5s), 80-90% Hungarian, survived ALL rounds",
        "context_length": 1048576,
        "supports_reasoning": True,
        "reasoning_type": "max_tokens",
        "recommended_for": ["hungarian", "speed", "lightweight", "battle_royale_champion"],
        "success_rate": 90,
        "avg_speed_seconds": 3.2,
    },
    {
        "id": "moonshotai/kimi-k2-0905:exacto",
        "name": "Kimi K2 (Exacto)",
        "description": "🥉 BALANCED CHAMPION - Fast (2-5s) with high quality (80-92%), survived ALL rounds",
        "context_length": 262144,
        "supports_reasoning": False,
        "reasoning_type": None,
        "recommended_for": ["hungarian", "balanced", "quality", "battle_royale_champion"],
        "success_rate": 92,
        "avg_speed_seconds": 4.0,
    },
    # 🎖️ QUALITY CHAMPIONS
    {
        "id": "google/gemini-2.5-flash-preview-09-2025",
        "name": "Gemini 2.5 Flash Preview",
        "description": "🎖️ QUALITY CHAMPION - Excellent quality (80-92%), medium speed (4-14s), survived ALL rounds",
        "context_length": 1048576,
        "supports_reasoning": True,
        "reasoning_type": "max_tokens",
        "recommended_for": ["hungarian", "quality", "reasoning", "battle_royale_champion"],
        "success_rate": 92,
        "avg_speed_seconds": 8.5,
    },
    {
        "id": "anthropic/claude-haiku-4.5",
        "name": "Claude Haiku 4.5",
        "description": "🎖️ QUALITY CHAMPION - Highest quality (80-93%), premium reliability, survived ALL rounds",
        "context_length": 200000,
        "supports_reasoning": True,
        "reasoning_type": "max_tokens",
        "recommended_for": ["hungarian", "premium", "quality", "battle_royale_champion"],
        "success_rate": 93,
        "avg_speed_seconds": 13.0,
    },
    {
        "id": "anthropic/claude-sonnet-4.5",
        "name": "Claude Sonnet 4.5",
        "description": "🎖️ PREMIUM CHAMPION - Excellent quality (80-92%), nuanced translation, survived ALL rounds",
        "context_length": 1000000,
        "supports_reasoning": True,
        "reasoning_type": "max_tokens",
        "recommended_for": ["hungarian", "premium", "nuanced", "battle_royale_champion"],
        "success_rate": 92,
        "avg_speed_seconds": 18.0,
    },
]

# EXCELLENT FREE MODELS - Survived ALL Battle Royale rounds at ZERO cost!
EXCELLENT_FREE_MODELS = [
    {
        "id": "amazon/nova-2-lite-v1:free",
        "name": "Amazon Nova 2 Lite (FREE)",
        "description": "🆓 FREE CHAMPION - Best free model! 80-95% Hungarian, 9-24s, survived ALL 6 rounds!",
        "context_length": 131072,
        "supports_reasoning": False,
        "reasoning_type": None,
        "recommended_for": ["hungarian", "free", "champion", "cost_efficient", "battle_royale_champion"],
        "success_rate": 95,
        "avg_speed_seconds": 17.0,
    },
    {
        "id": "nex-agi/deepseek-v3.1-nex-n1:free",
        "name": "NEX AGI DeepSeek V3.1 (FREE)",
        "description": "🆓 FREE SURVIVOR - Survived all rounds! 80-90% Hungarian, slower (10-64s) but reliable",
        "context_length": 163840,
        "supports_reasoning": False,
        "reasoning_type": None,
        "recommended_for": ["hungarian", "free", "reliable", "backup", "battle_royale_champion"],
        "success_rate": 90,
        "avg_speed_seconds": 35.0,
    },
]

# ELIMINATED MODELS - Failed Battle Royale Round 1
# DO NOT USE for Hungarian translation!
ELIMINATED_MODELS = [
    {
        "id": "tngtech/deepseek-r1t-chimera:free",
        "name": "TNG DeepSeek R1T (FREE)",
        "description": "❌ ELIMINATED Round 1 - 0% Hungarian output, DO NOT USE",
        "eliminated_round": 1,
        "reason": "0% Hungarian translation",
    },
    {
        "id": "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
        "name": "Dolphin Mistral (FREE)",
        "description": "❌ ELIMINATED Round 1 - 0% Hungarian output, DO NOT USE",
        "eliminated_round": 1,
        "reason": "0% Hungarian translation",
    },
    {
        "id": "nvidia/nemotron-3-nano-30b-a3b:free",
        "name": "NVIDIA Nemotron (FREE)",
        "description": "❌ ELIMINATED Round 1 - 0% Hungarian output, DO NOT USE",
        "eliminated_round": 1,
        "reason": "0% Hungarian translation",
    },
    {
        "id": "allenai/olmo-3-32b-think:free",
        "name": "Allen AI Olmo (FREE)",
        "description": "❌ ELIMINATED Round 1 - Timeout, too slow to compete",
        "eliminated_round": 1,
        "reason": "Timeout",
    },
    {
        "id": "openai/gpt-oss-120b:exacto",
        "name": "GPT OSS 120B (Exacto)",
        "description": "❌ ELIMINATED Round 1 - Timeout, too slow to compete",
        "eliminated_round": 1,
        "reason": "Timeout",
    },
]

# Empty PARTIAL_FREE_MODELS since we moved working ones to EXCELLENT
PARTIAL_FREE_MODELS = []

# GOOD: Secondary tier - not tested in Battle Royale
GOOD_MODELS = [
    {
        "id": "openai/gpt-oss-120b",
        "name": "GPT OSS 120B",
        "description": "SECONDARY - Not tested in Battle Royale but previously worked at 66%",
        "context_length": 131072,
        "supports_reasoning": True,
        "reasoning_type": "max_tokens",
        "recommended_for": ["hungarian", "backup"],
        "success_rate": 66,
    },
]

# POOR: These models struggle with Hungarian translation
POOR_MODELS = [
    {
        "id": "x-ai/grok-4.1-fast",
        "name": "Grok 4.1 Fast",
        "description": "POOR for Hungarian translation - Fails to produce complete translations even with reasoning enabled",
        "context_length": 2000000,
        "supports_reasoning": True,
        "reasoning_type": "enabled",
        "recommended_for": ["tool_calling", "general_but_not_hungarian"],
        "success_rate": 0,  # Based on testing
    },
    {
        "id": "meta-llama/llama-3.1-8b",
        "name": "Llama 3.1 8B",
        "description": "POOR for Hungarian translation - Too small for reliable translation",
        "context_length": 131072,
        "supports_reasoning": False,
        "recommended_for": ["speed", "simple_tasks"],
        "success_rate": 20,  # Based on expected performance
    },
    {
        "id": "google/gemma",  # Generic name for smaller models
        "name": "Gemma Models",
        "description": "POOR for Hungarian translation - Lightweight models not suitable for complex translation tasks",
        "context_length": 8192,
        "supports_reasoning": False,
        "recommended_for": ["lightweight_testing"],
        "success_rate": 10,  # Based on expected performance
    },
]

# Combine all successful models into main recommendations with priorities
RECOMMENDED_MODELS = EXCELLENT_MODELS + EXCELLENT_FREE_MODELS + GOOD_MODELS + PARTIAL_FREE_MODELS

# Testing reference for developers to run tests
TESTING_REFERENCE = {
    "excellent_models": ["anthropic/claude-haiku-4.5", "moonshotai/kimi-k2-0905:exacto"],
    "good_models": ["google/gemini-2.5-flash-preview-09-2025", "google/gemini-2.5-flash-lite-preview-09-2025",
                  "anthropic/claude-sonnet-4.5", "openai/gpt-oss-120b:exacto", "openai/gpt-oss-120b", "meta-llama/llama-4-maverick"],
    "avoid_models": ["x-ai/grok-4.1-fast", "minimax/minimax-m2", "deepseek/deepseek-v3.2-speciale", "z-ai/glm-4.6"],
    "hungarian_success_rate": {
        "claude_haiku_4.5": 100,
        "kimi_k2_exacto": 100,
        "gemini_flash": 66,
        "gemini_flash_lite": 66,
        "claude_sonnet_4.5": 66,
        "gpt_oss_120b": 66,
        "llama_4_maverick": 66
    }
}

# Anthropic models that require explicit cache_control
ANTHROPIC_MODELS = [
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.5-haiku",
    "anthropic/claude-3-opus",
]

# Models that support the :thinking variant
THINKING_VARIANT_MODELS = [
    "deepseek/deepseek-r1",
    "deepseek/deepseek-chat",
    "qwen/qwen3-vl-8b-thinking",
]

# Models that support reasoning with enabled flag (like Grok)
ENABLED_REASONING_MODELS = [
    "x-ai/grok-4.1-fast",
    "x-ai/grok-4.1",
]

# Models that support reasoning with effort levels (OpenAI o-series, GPT-5 series)
EFFORT_REASONING_MODELS = [
    "openai/o3-mini",
    "openai/o1",
    "openai/o1-mini",
    "openai/o4-mini",
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
]

# Models that support reasoning with max_tokens
MAX_TOKENS_REASONING_MODELS = [
    "google/gemini-2.5-flash-preview-09-2025",
    "google/gemini-2.5-flash-lite-preview-09-2025",
    "google/gemini-2.5-pro-preview",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-oss-120b",
    "minimax/minimax-m2",
    "deepseek/deepseek-v3.2-speciale",
    "z-ai/glm-4.6",
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
        Get list of recommended models for subtitle translation, specially curated for Hungarian.

        Returns:
            List of model information dictionaries with success rates and recommendations
        """
        default_model = self.settings.openrouter_default_model
        models = []
        
        # Prioritize the excellent models first
        for model in EXCELLENT_MODELS:
            model_info = model.copy()
            model_info["is_default"] = model["id"] == default_model
            model_info["priority"] = "excellent"
            models.append(model_info)
        
        # Add good models as secondary options
        for model in GOOD_MODELS:
            model_info = model.copy()
            model_info["is_default"] = model["id"] == default_model
            model_info["priority"] = "good"
            models.append(model_info)

        # Sort by priority (excellent first, then good), then by success rate within each category
        def sort_key(model):
            priority_score = 0 if model.get("priority") == "excellent" else 1
            
            # Excellent free models get highest priority (negative scores)
            if model.get("id") in ["allenai/olmo-3-32b-think:free", "tngtech/deepseek-r1t-chimera:free"]:
                priority_score = -2  # Excellent free models first
            elif model.get("id") in ["anthropic/claude-haiku-4.5", "moonshotai/kimi-k2-0905:exacto"]:
                priority_score = -1  # Existing excellent models second
            
            success_rate = model.get("success_rate", 0)
            
            # If default model, put it first regardless of priority
            if model.get("is_default", False):
                return (-100, -success_rate)
            return (priority_score, -success_rate)
        
        # Add excellent free models to the top of recommendations
        def sort_key(model):
            priority_score = 0 if model.get("priority") == "excellent" else 1
            if model.get("id") in [m["id"] for m in EXCELLENT_FREE_MODELS]:
                priority_score = -1  # Excellent free models get highest priority
            if model.get("id") in [m["id"] for m in EXCELLENT_MODELS]:
                priority_score = -2  # Existing excellent models get second priority
            success_rate = model.get("success_rate", 0)
            
            # If default model, put it first regardless of priority
            if model.get("is_default", False):
                return (-100, -success_rate)  # Default model always first
            return (priority_score, -success_rate)
        
        return sorted(models, key=sort_key)

    def get_best_model_for_language(self, target_language: str) -> Optional[str]:
        """
        Get the best model for the specified target language.
        
        Args:
            target_language: Target language code (e.g., 'hu' for Hungarian)
            
        Returns:
            Best model ID for the language, or None if no specific recommendations
        """
        target_lang = target_language.lower()
        
        # Check for Hungarian specifically
        if "hu" in target_lang or "hungarian" in target_lang:
            # Return the best performing Hungarian model
            if EXCELLENT_MODELS:
                return EXCELLENT_MODELS[0]["id"]
            elif GOOD_MODELS:
                return GOOD_MODELS[0]["id"]
        
        # For other languages, return reasoning-enabled models
        for model_info in RECOMMENDED_MODELS:
            if model_info.get("supports_reasoning"):
                return model_info["id"]
        
        # Fallback to first recommended model
        if RECOMMENDED_MODELS:
            return RECOMMENDED_MODELS[0]["id"]
        
        return None

    def get_hungarian_recommendations(self) -> dict[str, Any]:
        """
        Get specific recommendations for Hungarian subtitle translation.
        
        Returns:
            Dictionary with model recommendations and configurations for Hungarian
        """
        return {
            "best": EXCELLENT_MODELS,
            "also_good": GOOD_MODELS,
            "avoid": [model["id"] for model in POOR_MODELS],
            "recommended_config": {
                "reasoning": {"enabled": True},
                "temperature": 0.3,
            },
            "testing_reference": TESTING_REFERENCE
        }

    def _get_reasoning_type(self, model_id: str) -> Optional[str]:
        """
        Determine the reasoning type supported by a model.
        
        Args:
            model_id: The model identifier (without :thinking suffix)
            
        Returns:
            Reasoning type: 'thinking_variant', 'effort', 'max_tokens', 'enabled', or None
        """
        base_model = model_id.replace(":thinking", "")
        
        if base_model in THINKING_VARIANT_MODELS:
            return "thinking_variant"
        if base_model in ENABLED_REASONING_MODELS:
            return "enabled"
        if base_model in EFFORT_REASONING_MODELS:
            return "effort"
        if base_model in MAX_TOKENS_REASONING_MODELS:
            return "max_tokens"
        
        # Check RECOMMENDED_MODELS for reasoning type
        for model_info in RECOMMENDED_MODELS:
            if model_info["id"] == base_model:
                if model_info.get("supports_reasoning"):
                    return model_info.get("reasoning_type")
                return None
                
        return None

    def _build_reasoning_payload(
        self,
        model_id: str,
        config_override: Optional["TranslationConfig"],
    ) -> tuple[str, dict[str, Any]]:
        """
        Build reasoning-related payload parameters.
        
        Args:
            model_id: The base model identifier
            config_override: Optional config with reasoning settings
            
        Returns:
            Tuple of (final_model_id, reasoning_params_dict)
        """
        reasoning_params: dict[str, Any] = {}
        final_model_id = model_id
        
        if not config_override:
            return final_model_id, reasoning_params
            
        reasoning_config = config_override.reasoning
        use_thinking = config_override.use_thinking_variant
        
        # Check if model supports reasoning
        reasoning_type = self._get_reasoning_type(model_id)
        
        if reasoning_type is None:
            # Model doesn't support reasoning, skip
            if reasoning_config or use_thinking:
                logger.warning(
                    f"Model {model_id} does not support reasoning/thinking. "
                    "Reasoning settings will be ignored."
                )
            return final_model_id, reasoning_params
        
        # Handle :thinking variant
        if use_thinking and reasoning_type == "thinking_variant":
            if not model_id.endswith(":thinking"):
                final_model_id = f"{model_id}:thinking"
                logger.info(f"Using thinking variant: {final_model_id}")
            return final_model_id, reasoning_params
        
        # Handle reasoning config
        if reasoning_config:
            if reasoning_type == "enabled":
                # For models like Grok that use reasoning.enabled parameter
                if reasoning_config.enabled:
                    reasoning_params["reasoning"] = {"enabled": True}
                    logger.info("Using reasoning enabled: true")
                elif reasoning_config.enabled is False:
                    reasoning_params["reasoning"] = {"enabled": False}
                    logger.info("Using reasoning enabled: false")
                    
            elif reasoning_type == "effort":
                # Build effort-based reasoning params (OpenAI o-series)
                if reasoning_config.effort:
                    valid_efforts = ["xhigh", "high", "medium", "low", "minimal", "none"]
                    if reasoning_config.effort.lower() in valid_efforts:
                        reasoning_params["reasoning"] = {"effort": reasoning_config.effort.lower()}
                        logger.info(f"Using reasoning effort: {reasoning_config.effort}")
                    else:
                        logger.warning(f"Invalid reasoning effort: {reasoning_config.effort}")
                elif reasoning_config.enabled:
                    reasoning_params["reasoning"] = {"effort": "medium"}
                    logger.info("Using default reasoning effort: medium")
                    
            elif reasoning_type == "max_tokens":
                # Build max_tokens-based reasoning params (Gemini, Claude, MiniMax)
                if reasoning_config.max_tokens:
                    reasoning_params["reasoning"] = {"max_tokens": reasoning_config.max_tokens}
                    logger.info(f"Using reasoning max_tokens: {reasoning_config.max_tokens}")
                elif reasoning_config.enabled:
                    # Default to 2000 tokens for reasoning
                    reasoning_params["reasoning"] = {"max_tokens": 2000}
                    logger.info("Using default reasoning max_tokens: 2000")
                    
            elif reasoning_type == "thinking_variant":
                # For thinking variant models (DeepSeek, Qwen), use :thinking suffix
                if reasoning_config.enabled and not model_id.endswith(":thinking"):
                    final_model_id = f"{model_id}:thinking"
                    logger.info(f"Using thinking variant: {final_model_id}")
        
        return final_model_id, reasoning_params

    def _build_provider_payload(
        self,
        config_override: Optional["TranslationConfig"],
    ) -> dict[str, Any]:
        """
        Build provider routing configuration for OpenRouter.
        
        Args:
            config_override: Optional config with provider settings
            
        Returns:
            Provider configuration dict for the payload
        """
        provider_params: dict[str, Any] = {}
        
        if not config_override or not config_override.provider:
            # Default: prioritize throughput for fastest response
            return {"provider": {"sort": "throughput"}}
        
        provider_config = config_override.provider
        
        if provider_config.order:
            provider_params["order"] = provider_config.order
        
        if provider_config.allow_fallbacks is not None:
            provider_params["allow_fallbacks"] = provider_config.allow_fallbacks
            
        if provider_config.sort:
            provider_params["sort"] = provider_config.sort
        elif not provider_config.order:
            # Default to throughput sorting if no order specified
            provider_params["sort"] = "throughput"
            
        if provider_config.only:
            provider_params["only"] = provider_config.only
            
        if provider_config.ignore:
            provider_params["ignore"] = provider_config.ignore
        
        if provider_params:
            logger.debug(f"Provider routing params: {provider_params}")
            return {"provider": provider_params}
        
        return {}

    async def translate_batch(
        self,
        batch: TranslationBatch,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        config_override: Optional["TranslationConfig"] = None,
    ) -> TranslationResult:
        """
        Translate a batch of subtitle lines using OpenRouter.

        Args:
            batch: The batch of subtitle lines to translate
            model: Optional model override
            temperature: Optional temperature override
            config_override: Optional per-request configuration override

        Returns:
            TranslationResult containing translated lines and metadata

        Raises:
            TranslationProviderError: On translation failure
        """
        # Apply config override if provided (highest priority)
        if config_override:
            api_key = config_override.api_key or self.settings.openrouter_api_key
            model_to_use = config_override.model or model or self.settings.openrouter_default_model
            temp_to_use = (
                config_override.temperature
                if config_override.temperature is not None
                else (temperature if temperature is not None else self.settings.openrouter_temperature)
            )
            # Debug logging for API key tracking
            if config_override.api_key:
                logger.debug(f"Using API key from config_override: {config_override.api_key[:15]}...")
            else:
                logger.warning(f"config_override exists but api_key is None/empty. config_override fields: api_key={config_override.api_key is not None}, model={config_override.model}")
        else:
            api_key = self.settings.openrouter_api_key
            model_to_use = model or self.settings.openrouter_default_model
            temp_to_use = temperature if temperature is not None else self.settings.openrouter_temperature
            logger.debug("No config_override, using settings")

        # Validate API key
        if not api_key:
            raise TranslationProviderError(
                "OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable or pass apiKey in request config.",
                provider=self.provider_name,
                retryable=False,
            )

        # Build reasoning configuration
        model_to_use, reasoning_params = self._build_reasoning_payload(model_to_use, config_override)

        # Build messages
        system_prompt = self.build_system_prompt(
            batch.target_language,
            batch.source_language,
            batch.context_title,
            batch.context_media_type,
        )
        user_content = self.format_input_for_translation(batch.lines)

        # Build provider routing configuration
        provider_params = self._build_provider_payload(config_override)

        # Build messages with cache_control for Anthropic models
        is_anthropic = any(model_to_use.startswith(prefix) for prefix in ["anthropic/"])
        
        if is_anthropic:
            # Anthropic requires explicit cache_control breakpoints
            # Cache the system prompt since it's consistent across requests
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                },
                {"role": "user", "content": user_content},
            ]
        else:
            # Other providers use automatic caching
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

        # Build request payload
        payload: dict[str, Any] = {
            "model": model_to_use,
            "messages": messages,
            "temperature": temp_to_use,
            "max_tokens": self.settings.openrouter_max_tokens,
            "response_format": {"type": "json_object"},
            # Include usage stats to track cache savings
            "usage": {"include": True},
        }
        
        # Add reasoning params if configured
        if reasoning_params:
            payload.update(reasoning_params)
            
        # Add provider routing params
        if provider_params:
            payload.update(provider_params)

        # Debug logging: Log incoming batch data
        debug_logger.debug(f"=== INCOMING BATCH DATA ===")
        debug_logger.debug(f"Source language: {batch.source_language}")
        debug_logger.debug(f"Target language: {batch.target_language}")
        debug_logger.debug(f"Lines count: {len(batch.lines)}")
        debug_logger.debug(f"Input lines: {json.dumps(batch.lines, ensure_ascii=False, indent=2)}")
        
        logger.info(f"Sending translation request for {len(batch.lines)} lines using {model_to_use}")
        if reasoning_params:
            logger.debug(f"Reasoning params: {reasoning_params}")
        
        # Debug logging: Log the complete payload being sent to OpenRouter
        debug_logger.debug(f"=== SENDING TO OPENROUTER ===")
        debug_logger.debug(f"Model: {model_to_use}")
        debug_logger.debug(f"Temperature: {temp_to_use}")
        debug_logger.debug(f"Payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")

        try:
            # Create request-specific client if API key differs from default
            # Always use config_override API key if available
            if config_override and config_override.api_key:
                headers = self.settings.get_openrouter_headers(api_key_override=config_override.api_key)
                logger.info(f"Making request with per-request API key: {config_override.api_key[:15]}... Auth header set: {'Authorization' in headers}")
                async with httpx.AsyncClient(
                    base_url=self.settings.openrouter_api_base,
                    headers=headers,
                    timeout=httpx.Timeout(self.settings.request_timeout),
                ) as client:
                    response = await client.post("/chat/completions", json=payload)
                    result = await self._process_response(response, model_to_use)
                    # Perform safety validation
                    self._validate_and_warn_unchanged(batch.lines, result.translations)
                    return result
            else:
                logger.info(f"Making request with default client (env API key configured: {bool(self.settings.openrouter_api_key)})")
                response = await self.client.post("/chat/completions", json=payload)
                result = await self._process_response(response, model_to_use)
                # Perform safety validation
                self._validate_and_warn_unchanged(batch.lines, result.translations)
                return result
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

        # Debug logging: Log the raw response content
        debug_logger.debug(f"=== RECEIVED FROM OPENROUTER ===")
        debug_logger.debug(f"Model used: {model_used}")
        debug_logger.debug(f"Token usage - prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}")
        debug_logger.debug(f"Raw content: {content}")
        
        # Parse the JSON array from content
        translations = self._parse_translations(content)
        
        # Debug logging: Log parsed translations
        debug_logger.debug(f"=== PARSED TRANSLATIONS ===")
        debug_logger.debug(f"Translations count: {len(translations)}")
        debug_logger.debug(f"Translations: {json.dumps(translations, ensure_ascii=False, indent=2)}")

        return TranslationResult(
            translations=translations,
            model_used=model_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw_response=data,
        )
    
    def _validate_and_warn_unchanged(
        self,
        original_lines: list[dict[str, str]],
        translations: list[dict[str, str]],
    ) -> None:
        """
        Validate that translations are actually different from the original text.
        
        This is a safety check to detect when the model returns the original text
        instead of actually translating it. Logs warnings but does not fail.
        
        Args:
            original_lines: Original lines sent for translation
            translations: Translations received from the model
        """
        # Build lookup maps
        original_map: dict[str, str] = {str(line["index"]): line["content"] for line in original_lines}
        
        unchanged_indices: list[int] = []
        for trans in translations:
            idx = str(trans["index"])
            translated_content = trans["content"]
            original_content = original_map.get(idx, "")
            
            # Check if the content is exactly the same (potential untranslated)
            if translated_content.strip() == original_content.strip():
                try:
                    unchanged_indices.append(int(idx))
                except ValueError:
                    # If index is not an integer, just log it
                    pass
                debug_logger.warning(
                    f"⚠️ INDEX {idx}: Translation identical to original! "
                    f"Original: '{original_content[:50]}...' -> "
                    f"Translated: '{translated_content[:50]}...'"
                )
        
        # Calculate the percentage of unchanged translations
        if len(translations) > 0:
            unchanged_pct = (len(unchanged_indices) / len(translations)) * 100
            if unchanged_pct >= 50:
                logger.warning(
                    f"⚠️ SAFETY CHECK: {unchanged_pct:.1f}% of translations are identical to original! "
                    f"This may indicate the model is not translating properly. "
                    f"Unchanged indices: {unchanged_indices[:10]}{'...' if len(unchanged_indices) > 10 else ''}"
                )
            elif unchanged_pct >= 20:
                logger.info(
                    f"ℹ️ {unchanged_pct:.1f}% of translations are identical to original (may be intentional for names/numbers). "
                    f"Count: {len(unchanged_indices)}"
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