
from typing import Optional
from enum import Enum
from pydantic import Field
from pydantic_settings import BaseSettings

class ModelProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"

class Settings(BaseSettings):
    model_config = {"env_file": ".env"}

    benchmark_host: str = Field(..., env="BENCHMARK_HOST")
    benchmark_id: str = Field(..., env="BENCHMARK_ID")
    model_provider: ModelProvider = Field(ModelProvider.OPENAI, env="MODEL_PROVIDER")
    model_name: str = Field("gpt-5.4-mini", env="MODEL_NAME")
    model_api_token: Optional[str] = Field(None, env="MODEL_API_TOKEN")
    model_think: str = Field("high", env="MODEL_THINK")