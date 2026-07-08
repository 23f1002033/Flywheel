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
    fireworks_model: str = "accounts/fireworks/models/glm-5p2"

    local_base_url: str = ""
    local_model: str = "google/gemma-2-2b-it"

    cloud_price_input: float = 0.90
    cloud_price_output: float = 0.90
    frontier_price_input: float = 5.00    
    frontier_price_output: float = 15.00

    # Router Agent
    complexity_threshold: float = 2.0
    pin_sensitive_local: bool = True

    # Memory + cost engine (M4)
    db_url: str = "sqlite:///data/flywheel.db"
    co2_grams_per_1k_cloud_tokens: float = 1.5   # rough industry estimate

    # Semantic cache + learning router + budget (M5)
    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.95
    memory_routing_enabled: bool = True
    memory_k: int = 8
    memory_min_samples: int = 4
    monthly_budget_usd: float = 50.0

@lru_cache
def get_settings() -> Settings:
    return Settings()