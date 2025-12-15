"""Configuration management using pydantic-settings."""

from functools import lru_cache
from typing import Optional

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
    openrouter_default_model: str = "google/gemini-2.5-flash-preview-05-20"
    openrouter_temperature: float = 0.3
    openrouter_max_tokens: int = 8000

    # Translation Configuration
    batch_size: int = 100
    max_retries: int = 3
    retry_delay: float = 1.0
    request_timeout: float = 120.0

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

    @property
    def openrouter_headers(self) -> dict[str, str]:
        """Get headers for OpenRouter API requests."""
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "X-Title": self.app_name,
        }
        if self.app_url:
            headers["HTTP-Referer"] = self.app_url
        return headers

    def is_rtl_language(self, language_code: str) -> bool:
        """Check if a language code represents an RTL language."""
        # Handle full language names or codes
        code = language_code.lower().split("-")[0].split("_")[0]
        return code in self.rtl_languages


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()