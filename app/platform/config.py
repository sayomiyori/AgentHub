from functools import lru_cache
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PlatformSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        hide_input_in_errors=True,
        populate_by_name=True,
    )
    telegram_ai_enabled: bool = Field(default=False, alias="TELEGRAM_AI_ENABLED")
    provider: Literal["groq"] = Field(default="groq", alias="TELEGRAM_AI_PROVIDER")
    model: str = Field(default="", alias="TELEGRAM_AI_MODEL", max_length=128)
    webhook_url: str = Field(default="", alias="WEBHOOK_INTERNAL_URL")
    ingress_key: SecretStr = Field(default=SecretStr(""), alias="WEBHOOK_AGENT_INGRESS_KEY")
    context_key: SecretStr = Field(default=SecretStr(""), alias="WEBHOOK_AGENT_SERVICE_KEY")
    reply_key: SecretStr = Field(default=SecretStr(""), alias="AGENT_WEBHOOK_REPLY_KEY")

    @model_validator(mode="after")
    def validate_enabled(self) -> "PlatformSettings":
        if not self.telegram_ai_enabled:
            return self
        keys = [
            x.get_secret_value()
            for x in (
                self.ingress_key,
                self.context_key,
                self.reply_key,
            )
        ]
        if any(len(x) < 32 or not x.isascii() or not x.isprintable() for x in keys):
            raise ValueError("Independent platform service keys are required")
        if len(set(keys)) != 3:
            raise ValueError("Platform service keys must be independent")
        url = urlsplit(self.webhook_url)
        port = url.port  # Access validates nonnumeric and out-of-range ports.
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.netloc.endswith(":")
            or port == 0
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
        ):
            raise ValueError("A fixed internal WebHook origin is required")
        if not self.model.strip() or self.model != self.model.strip():
            raise ValueError("An explicit platform model is required")
        return self


@lru_cache(maxsize=1)
def get_platform_settings() -> PlatformSettings:
    return PlatformSettings()
