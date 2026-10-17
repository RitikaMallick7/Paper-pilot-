"""Module to chunk text by logical sections."""
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """A chunk of text with metadata."""
    text: str
    paper_title: str
    arxiv_id: str
    authors: List[str]
    section: str
    published: str
    chunk_id: str


class SectionChunker:
    """Chunks text by logical sections with paragraph-level fallback."""
    
    def __init__(self, max_chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the chunker.
        
        Args:
            max_chunk_size: Maximum characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_sections(
        self,
        sections: Dict[str, str],
        metadata: Dict
    ) -> List[DocumentChunk]:
        """
        Chunk each section into smaller pieces if needed.
        
        Trade-off: Using character-based chunking instead of token-based for simplicity.
        Token-based would be more accurate for LLM context but requires tokenizer.
        
        Args:
            sections: Dictionary of section names to content
            metadata: Paper metadata (title, arxiv_id, authors, published)
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        chunk_counter = 0
        
        for section_name, section_content in sections.items():
            # Skip empty sections
            if not section_content or len(section_content.strip()) == 0:
                continue
            
            # If section is small enough, keep it as one chunk
            if len(section_content) <= self.max_chunk_size:
                chunk = DocumentChunk(
                    text=section_content,
                    paper_title=metadata["title"],
                    arxiv_id=metadata["arxiv_id"],
                    authors=metadata["authors"],
                    section=section_name,
                    published=metadata["published"],
                    chunk_id=f"{metadata['arxiv_id']}_{section_name}_{chunk_counter}"
                )
                chunks.append(chunk)
                chunk_counter += 1
            else:
                # Split large sections into overlapping chunks
                section_chunks = self._split_text(section_content)
                for i, chunk_text in enumerate(section_chunks):
                    chunk = DocumentChunk(
                        text=chunk_text,
                        paper_title=metadata["title"],
                        arxiv_id=metadata["arxiv_id"],
                        authors=metadata["authors"],
                        section=section_name,
                        published=metadata["published"],
                        chunk_id=f"{metadata['arxiv_id']}_{section_name}_{chunk_counter}"
                    )
                    chunks.append(chunk)
                    chunk_counter += 1
        
        return chunks
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks by character count.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.max_chunk_size
            
            # Try to break at a paragraph boundary
            if end < len(text):
                # Look for the last newline before the end
                last_newline = text.rfind('\n', start, end)
                if last_newline > start + self.max_chunk_size // 2:
                    end = last_newline + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
        
        return chunks
