
from typing import Optional
from enum import Enum
from pydantic import BaseSettings, Field

class ModelProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"

class Settings(BaseSettings):
    benchmark_host = Field(..., env="BENCHMARK_HOST")
    benchmark_id = Field(..., env="BENCHMARK_ID")
    MODEL_PROVIDER: ModelProvider = Field(ModelProvider.OPENAI, env="MODEL_PROVIDER")
    MODEL_NAME: str = Field("gpt-5.4-mini", env="MODEL_NAME")
    MODEL_API_TOKEN: Optional[str] = Field(None, env="MODEL_API_TOKEN")