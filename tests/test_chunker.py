"""Tests for the chunker module."""
from ingestion.chunker import SectionChunker, DocumentChunk


def test_chunker_initialization():
    """Test that chunker initializes with default parameters."""
    chunker = SectionChunker()
    assert chunker.max_chunk_size == 1000
    assert chunker.chunk_overlap == 200


def test_chunker_custom_params():
    """Test that chunker accepts custom parameters."""
    chunker = SectionChunker(max_chunk_size=500, chunk_overlap=100)
    assert chunker.max_chunk_size == 500
    assert chunker.chunk_overlap == 100


def test_chunk_sections():
    """Test section-based chunking."""
    chunker = SectionChunker(max_chunk_size=100, chunk_overlap=20)
    
    sections = {
        "Abstract": "This is an abstract. It describes the paper.",
        "Introduction": "This is the introduction. It provides background. It sets up the problem."
    }
    
    metadata = {
        "title": "Test Paper",
        "arxiv_id": "test123",
        "authors": ["Author One", "Author Two"],
        "published": "2024-01-01"
    }
    
    chunks = chunker.chunk_sections(sections, metadata)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
    assert all(chunk.paper_title == "Test Paper" for chunk in chunks)
