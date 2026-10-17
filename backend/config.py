"""Configuration management from environment variables."""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    
    # LLM Provider
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic")
    
    # API Keys
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Qdrant Configuration
   # 
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    
    # Ingestion Configuration
    ARXIV_CATEGORY: str = os.getenv("ARXIV_CATEGORY", "cs.AI")
    NUM_PAPERS_TO_INGEST: int = int(os.getenv("NUM_PAPERS_TO_INGEST", "10"))
    
    # Embedding Model
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    
    # Collection name in Qdrant
    COLLECTION_NAME: str = "paper_chunks"


config = Config()
