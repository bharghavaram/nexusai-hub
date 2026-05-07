"""NexusAI Hub — Central Configuration."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    APP_NAME: str = "NexusAI Hub"
    APP_VERSION: str = "1.0.0"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = False

    # LLM
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o"
    CLAUDE_MODEL: str = "claude-3-5-sonnet-20241022"
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 2048

    # SLM (HuggingFace)
    HF_MODEL_SENTIMENT: str = "distilbert-base-uncased-finetuned-sst-2-english"
    HF_MODEL_NER: str = "dslim/bert-base-NER"
    HF_MODEL_SUMMARISE: str = "facebook/bart-large-cnn"
    HF_MODEL_ZERO_SHOT: str = "facebook/bart-large-mnli"
    SLM_MAX_LENGTH: int = 512
    SLM_DEVICE: str = "cpu"

    # RAG / Vector
    FAISS_INDEX_PATH: str = "data/faiss_index"
    EMBED_MODEL: str = "text-embedding-3-small"
    RAG_TOP_K: int = 5

    # Automation
    MAX_WORKFLOW_STEPS: int = 20
    EXECUTION_TIMEOUT_SECS: int = 300
    MAX_PARALLEL_TASKS: int = 5

    # GenAI
    GENAI_MAX_TOKENS: int = 4096
    GENAI_TEMPERATURE: float = 0.7

    # Storage
    DATA_DIR: str = "data"
    OUTPUT_DIR: str = "outputs"
    DB_URL: str = "sqlite:///nexusai.db"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
