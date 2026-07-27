import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "your-gemini-key")
    UPLOAD_FOLDER: str = "./data/raw_documents"
    VECTOR_DB_DIR: str = "./data/vector_db"
    MODEL_PATH: str = "./models/tf_classifier.h5"
    DATABASE_URL: str = "sqlite:///./data/metadata.db"

    class Config:
        env_file = ".env"

settings = Settings()