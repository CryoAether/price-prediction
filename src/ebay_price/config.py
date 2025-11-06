from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    MLFLOW_TRACKING_URI: str = Field(default="file:./data/artifacts/mlruns")
    MLFLOW_EXPERIMENT: str = Field(default="price-prediction")


settings = Settings()
