from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime config comes from environment / .env - never hardcode."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="FLYWHEEL_", extra="ignore")

    # App
    env: str = "dev"                      
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    fireworks_api_key: str = ""
    fireworks_base_url: str = "https://api.fireworks.ai/inference/v1"
    fireworks_model: str = "accounts/fireworks/models/llama-v3p1-70b-instruct"

    local_base_url: str = ""
    local_model: str = "google/gemma-2-2b-it"

    cloud_price_input: float = 0.90
    cloud_price_output: float = 0.90
    frontier_price_input: float = 5.00    
    frontier_price_output: float = 15.00


@lru_cache
def get_settings() -> Settings:
    return Settings()