from enum import Enum
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


class ModelThinking(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    BENCHMARK_HOST: str = Field("https://api.bitgn.com")
    BENCHMARK_ID: str = Field("bitgn/pac1-dev")
    MODEL_PROVIDER: ModelProvider = Field(ModelProvider.OPENAI)
    MODEL_NAME: str = Field("gpt-5.4")
    AZURE_OPENAI_API_KEY: str = Field("")
    AZURE_OPENAI_ENDPOINT: str = Field("")
    MODEL_THINKING: Optional[ModelThinking] = Field(None)
    MAX_PARALLEL_TASKS: int = Field(5, ge=1, le=5)
    BENCH_API_KEY: str
