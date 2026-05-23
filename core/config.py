"""
Configuration settings for the ai_call_agent.
Loads settings from environment variables using Pydantic Settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Settings class containing all environment configuration options for the application.
    Automatically loads variables from .env file or system environment.
    """
    # Twilio Configuration
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_phone_number: str
    human_agent_number: str

    # Deepgram Configuration
    deepgram_api_key: str

    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "nemotron-3-super:cloud"

    # ChromaDB Configuration
    chroma_persist_dir: str = "./rag/chroma_db"
    chroma_collection_name: str = "hospital_faqs"

    # Embeddings Configuration
    embedding_model: str = "nomic-embed-text-v2-moe"

    # Mock API Configuration
    mock_api_base_url: str = "http://localhost:5000"
    appointments_db_path: str = "./mock_api/appointments.db"

    # LangSmith Configuration
    langchain_api_key: str
    langchain_project: str = "ai_call_agent"
    langchain_tracing_v2: str = "true"

    # App Configuration
    app_host: str = "0.0.0.0"
    app_port: int = 5000
    websocket_path: str = "/twilio"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

# Expose a module-level singleton instance of Settings
settings = Settings()

import logging
logger = logging.getLogger(__name__)

def configure_langsmith() -> None:
    """
    Configure LangSmith tracing environment variables.
    Must be called before importing any LangChain/LangGraph modules.
    """
    import os
    os.environ["LANGCHAIN_TRACING_V2"] = settings.langchain_tracing_v2
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    logger.info(f"LangSmith tracing configured for project: {settings.langchain_project}")
