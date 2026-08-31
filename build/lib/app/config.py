from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GATEWAY_", env_file=".env", extra="ignore")

    # Comma-separated client keys. Empty disables auth (dev only).
    api_keys: str = ""
    # Not used yet: this is where the token rate limiter will keep its buckets.
    redis_url: str = ""

    @property
    def allowed_keys(self) -> set[str]:
        return {key.strip() for key in self.api_keys.split(",") if key.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
