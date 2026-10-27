import re
from functools import lru_cache
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MODEL_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}")


class PlatformSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        hide_input_in_errors=True,
        populate_by_name=True,
    )
    telegram_ai_enabled: bool = Field(default=False, alias="TELEGRAM_AI_ENABLED")
    read_enabled: bool = Field(default=False, alias="PLATFORM_READ_ENABLED")
    tenant_enabled: bool = Field(default=False, alias="PLATFORM_TENANT_ENABLED")
    auth_url: str = Field(default="", alias="AUTHFORTRESS_BASE_URL")
    provider: Literal["groq"] = Field(default="groq", alias="TELEGRAM_AI_PROVIDER")
    model: str = Field(default="", alias="TELEGRAM_AI_MODEL", max_length=128)
    webhook_url: str = Field(default="", alias="WEBHOOK_INTERNAL_URL")
    ingress_key: SecretStr = Field(default=SecretStr(""), alias="WEBHOOK_AGENT_INGRESS_KEY")
    context_key: SecretStr = Field(default=SecretStr(""), alias="WEBHOOK_AGENT_SERVICE_KEY")
    reply_key: SecretStr = Field(default=SecretStr(""), alias="AGENT_WEBHOOK_REPLY_KEY")

    @model_validator(mode="after")
    def validate_enabled(self) -> "PlatformSettings":
        if not self.telegram_ai_enabled and not self.read_enabled and not self.tenant_enabled:
            return self
        keys = [
            x.get_secret_value()
            for x in ((
                self.ingress_key,
                self.context_key,
                self.reply_key,
            ) if self.telegram_ai_enabled else ((self.context_key,) if self.read_enabled else ()))
        ]
        if any(len(x) < 32 or not x.isascii() or not x.isprintable() for x in keys):
            raise ValueError("Independent platform service keys are required")
        if len(set(keys)) != len(keys):
            raise ValueError("Platform service keys must be independent")
        origins = [self.webhook_url] if self.telegram_ai_enabled or self.read_enabled else []
        if self.read_enabled or self.tenant_enabled:
            origins.append(self.auth_url)
        for origin in origins:
            url = urlsplit(origin)
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
                or not origin.isascii()
                or any(c.isspace() or ord(c) < 32 for c in origin)
            ):
                raise ValueError("A fixed internal origin is required")
        if self.telegram_ai_enabled and not MODEL_IDENTIFIER.fullmatch(self.model):
            raise ValueError("An explicit platform model is required")
        return self


@lru_cache(maxsize=1)
def get_platform_settings() -> PlatformSettings:
    return PlatformSettings()
