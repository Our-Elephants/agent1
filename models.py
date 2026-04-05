from enum import Enum
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    BENCHMARK_HOST: str = Field("https://api.bitgn.com")
    BENCHMARK_ID: str = Field("bitgn/pac1-dev")
    MODEL_PROVIDER: ModelProvider = Field(ModelProvider.OPENAI)
    MODEL_NAME: str = Field("gpt-5.4-mini")
    MODEL_API_TOKEN: str = Field("")
    MODEL_THINKING: Optional[str] = Field(None)
