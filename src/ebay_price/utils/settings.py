from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Sections
    ebay__app_id: str = Field(default="")
    ebay__client_id: str = Field(default="")
    ebay__client_secret: str = Field(default="")
    ebay__site: str = Field(default="EBAY_US")

    storage__data_root: str = Field(default="data")
    storage__duckdb_path: str = Field(default="data/artifacts/warehouse.duckdb")
    storage__parquet_bucket: str = Field(default="data/raw")

    postgres__url: str = Field(
        default="postgresql+psycopg://user:password@localhost:5432/ebay_price"
    )
    postgres__schema: str = Field(default="public")


def load_settings() -> AppSettings:
    # Load layered: .env variables override TOML on disk
    # Optional TOML loader for local settings file
    s = AppSettings()
    return s
