"""Settings the translation core reads. Plain values; nothing here touches the environment."""

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_RTL_LANGUAGES = ["ar", "he", "fa", "ur", "yi", "ps", "sd", "ku"]


class CoreSettings(BaseModel):
    """Everything the core and the back ends read.

    The server builds one from its environment-backed Settings; the Bazarr+ plugin and any
    library consumer build one directly. Unknown fields are rejected so a server-only
    setting cannot be passed in by mistake and silently ignored.
    """

    model_config = ConfigDict(extra="forbid")

    openrouter_api_key: str = Field("", repr=False)
    openrouter_api_base: str = "https://openrouter.ai/api/v1"
    openrouter_default_model: str = "amazon/nova-2-lite-v1:free"
    openrouter_temperature: float = 0.3
    openrouter_max_tokens: int = 8000

    batch_size: int = 100
    parallel_batches_per_job: int = 4
    max_retries: int = 3
    retry_delay: float = 1.0
    request_timeout: float = 120.0

    app_name: str = "ai-subtitle-translator"
    app_url: str | None = "https://lavx.hu"

    rtl_languages: list[str] = Field(default_factory=lambda: list(DEFAULT_RTL_LANGUAGES))

    def get_openrouter_headers(self, api_key_override: str | None = None) -> dict[str, str]:
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
        return self.get_openrouter_headers()

    def is_rtl_language(self, language_code: str) -> bool:
        code = language_code.lower().split("-")[0].split("_")[0]
        return code in self.rtl_languages
