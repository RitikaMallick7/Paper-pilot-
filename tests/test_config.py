"""Tests for configuration management."""
import pytest
from backend.config import config, Config


def test_config_initialization():
    """Test that config is properly initialized."""
    assert isinstance(config, Config)
    assert config.COLLECTION_NAME == "paper_chunks"


def test_config_defaults():
    """Test that config has sensible defaults."""
    assert config.QDRANT_HOST == "localhost"
    assert config.QDRANT_PORT == 6333
    assert config.EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"
