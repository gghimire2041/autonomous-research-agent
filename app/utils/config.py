"""Configuration management using Pydantic settings."""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )
    
    # Application
    DEBUG: bool = Field(False, description="Enable debug mode")
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    
    # LLM Configuration
    LLM_PROVIDER: str = Field("openai", description="LLM provider (openai, huggingface)")
    MODEL_NAME: str = Field("gpt-4-turbo-preview", description="Model name")
    OPENAI_API_KEY: Optional[str] = Field(None, description="OpenAI API key")
    HUGGINGFACE_API_KEY: Optional[str] = Field(None, description="HuggingFace API key")
    MAX_TOKENS: int = Field(2000, description="Maximum tokens in LLM response")
    TEMPERATURE: float = Field(0.1, description="LLM temperature")
    
    # Safety Settings
    SAFE_MODE: bool = Field(True, description="Enable safety guardrails")
    MAX_STEPS: int = Field(10, description="Maximum agent execution steps")
    TIMEOUT_SECONDS: int = Field(300, description="Maximum task execution time")
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = Field(100, description="Rate limit requests per window")
    RATE_LIMIT_WINDOW: int = Field(3600, description="Rate limit window in seconds")
    
    # Database
    DATABASE_URL: str = Field("sqlite:///./agent_memory.db", description="Database URL")
    
    # File Operations
    SANDBOX_DIR: str = Field("./sandbox", description="Sandboxed directory for file operations")
    MAX_FILE_SIZE: int = Field(10 * 1024 * 1024, description="Maximum file size (10MB)")
    
    # URL Allowlist (for safe mode)
    ALLOWED_DOMAINS: List[str] = Field(
        default=[
            "wikipedia.org",
            "github.com",
            "stackoverflow.com",
            "python.org",
            "docs.python.org",
            "arxiv.org",
            "scholar.google.com"
        ],
        description="Allowed domains for web fetching in safe mode"
    )
    
    # Observability
    ENABLE_METRICS: bool = Field(True, description="Enable Prometheus metrics")
    ENABLE_TRACING: bool = Field(True, description="Enable OpenTelemetry tracing")
    JAEGER_ENDPOINT: Optional[str] = Field(None, description="Jaeger tracing endpoint")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


