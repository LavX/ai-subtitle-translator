"""Configuration management using pydantic-settings."""

import threading
from datetime import timedelta
from typing import Any, Dict, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8765
    debug: bool = False

    # OpenRouter Configuration
    openrouter_api_key: str = ""
    openrouter_api_base: str = "https://openrouter.ai/api/v1"
    openrouter_default_model: str = "amazon/nova-2-lite-v1:free"
    openrouter_temperature: float = 0.3
    openrouter_max_tokens: int = 8000

    # Translation Configuration
    batch_size: int = 100
    parallel_batches_per_job: int = 4  # Number of batches to process in parallel per job
    max_retries: int = 3
    retry_delay: float = 1.0
    request_timeout: float = 120.0

    # Job Queue Configuration
    job_queue_max_concurrent: int = 15  # Max concurrent translation jobs (increased for battle royale)
    job_queue_max_jobs: int = 500  # Max jobs in memory (increased for high-throughput testing)
    job_queue_ttl_hours: int = 1  # TTL for completed/failed jobs

    # App identification for OpenRouter analytics
    app_name: str = "ai-subtitle-translator"
    app_url: Optional[str] = "https://lavx.hu"

    # RTL (Right-to-Left) language codes
    rtl_languages: list[str] = [
        "ar",  # Arabic
        "he",  # Hebrew
        "fa",  # Persian/Farsi
        "ur",  # Urdu
        "yi",  # Yiddish
        "ps",  # Pashto
        "sd",  # Sindhi
        "ku",  # Kurdish (Sorani)
    ]

    def get_openrouter_headers(self, api_key_override: Optional[str] = None) -> dict[str, str]:
        """
        Get headers for OpenRouter API requests.
        
        Args:
            api_key_override: Optional API key to use instead of the configured one
            
        Returns:
            Headers dictionary for OpenRouter API requests
        """
        api_key = api_key_override or self.openrouter_api_key
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": self.app_name,
        }
        if self.app_url:
            headers["HTTP-Referer"] = self.app_url
        return headers

    @property
    def openrouter_headers(self) -> dict[str, str]:
        """Get headers for OpenRouter API requests (legacy property)."""
        return self.get_openrouter_headers()

    def is_rtl_language(self, language_code: str) -> bool:
        """Check if a language code represents an RTL language."""
        # Handle full language names or codes
        code = language_code.lower().split("-")[0].split("_")[0]
        return code in self.rtl_languages

    @property
    def job_queue_ttl(self) -> timedelta:
        """Get job TTL as timedelta."""
        return timedelta(hours=self.job_queue_ttl_hours)


# Global settings instance with lock for thread safety
_settings: Optional[Settings] = None
_settings_lock = threading.Lock()
_runtime_overrides: Dict[str, Any] = {}


def get_settings() -> Settings:
    """
    Get settings instance with runtime overrides applied.
    
    This function returns a settings instance that combines:
    1. Environment variables / .env file values (base)
    2. Runtime overrides set via update_runtime_config
    
    Returns:
        Settings instance with runtime overrides applied
    """
    global _settings
    with _settings_lock:
        if _settings is None:
            _settings = Settings()
        
        # Apply runtime overrides
        if _runtime_overrides:
            # Create a copy of current settings with overrides
            settings_dict = _settings.model_dump()
            settings_dict.update(_runtime_overrides)
            return Settings(**settings_dict)
        
        return _settings


def update_runtime_config(key: str, value: Any) -> None:
    """
    Update configuration at runtime without restarting.
    
    This allows dynamic configuration changes through the API.
    The changes are stored in memory and will be lost on restart.
    
    Args:
        key: Configuration key to update (must be a valid Settings field)
        value: New value for the configuration
        
    Raises:
        ValueError: If key is not a valid Settings field
    """
    global _runtime_overrides
    
    # Validate that key is a valid Settings field
    valid_keys = set(Settings.model_fields.keys())
    if key not in valid_keys:
        raise ValueError(f"Invalid configuration key: {key}. Valid keys: {valid_keys}")
    
    with _settings_lock:
        _runtime_overrides[key] = value


def get_runtime_overrides() -> Dict[str, Any]:
    """
    Get current runtime configuration overrides.
    
    Returns:
        Dictionary of runtime overrides (keys are masked for sensitive values)
    """
    with _settings_lock:
        result = {}
        for key, value in _runtime_overrides.items():
            if "key" in key.lower() or "secret" in key.lower() or "password" in key.lower():
                result[key] = "***" if value else None
            else:
                result[key] = value
        return result


def clear_runtime_overrides() -> None:
    """Clear all runtime configuration overrides."""
    global _runtime_overrides
    with _settings_lock:
        _runtime_overrides = {}


def reset_settings() -> None:
    """Reset settings instance to force reload from environment."""
    global _settings, _runtime_overrides
    with _settings_lock:
        _settings = None
        _runtime_overrides = {}