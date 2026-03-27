from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AgentHub"
    debug: bool = False

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    llm_provider: str = Field(default="gemini", alias="LLM_PROVIDER")
    llm_model: str = Field(default="models/gemini-2.5-flash", alias="LLM_MODEL")
    llm_fallback_provider: str = Field(default="", alias="LLM_FALLBACK_PROVIDER")
    llm_fallback_model: str = Field(default="", alias="LLM_FALLBACK_MODEL")
    embedding_model: str = Field(default="models/embedding-001", alias="EMBEDDING_MODEL")

    # JSON array: [{"name":"github","url":"https://..."}, ...] or leave empty
    mcp_servers: str = Field(default="", alias="MCP_SERVERS")
    mcp_servers_config: str = Field(default="", alias="MCP_SERVERS_CONFIG")

    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@postgres:5432/agenthub",
        alias="DATABASE_URL",
    )
    redis_url: str = Field(default="redis://redis:6379/0", alias="REDIS_URL")
    semantic_cache_enabled: bool = Field(default=True, alias="SEMANTIC_CACHE_ENABLED")

    upload_dir: str = "./uploads"
    max_preview_chars: int = 220


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
