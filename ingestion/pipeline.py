"""Main ingestion pipeline orchestrator."""
import os
from typing import List
from ingestion.arxiv_fetcher import ArxivFetcher, PaperMetadata
from ingestion.pdf_parser import PDFParser
from ingestion.chunker import SectionChunker, DocumentChunk
from ingestion.embeddings import EmbeddingGenerator
from ingestion.vector_store import QdrantVectorStore
from backend.config import config


class IngestionPipeline:
    """End-to-end ingestion pipeline for arXiv papers."""
    
    def __init__(self):
        """Initialize the pipeline with all components."""
        self.fetcher = ArxivFetcher(
            category=config.ARXIV_CATEGORY,
            max_results=config.NUM_PAPERS_TO_INGEST
        )
        
        self.vector_store = QdrantVectorStore(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            collection_name=config.COLLECTION_NAME,
            embedding_dim=self.embedder.embedding_dim
        )
        self.papers_dir = "papers"
    
    def run(self, reingest: bool = False) -> int:
        """
        Run the full ingestion pipeline.
        
        Args:
            reingest: If True, delete existing data for papers before re-ingesting
            
        Returns:
            Number of chunks ingested
        """
        # Fetch paper metadata
        print(f"Fetching papers from category {config.ARXIV_CATEGORY}...")
        papers = self.fetcher.fetch_papers()
        print(f"Found {len(papers)} papers")
        
        total_chunks = 0
        
        for paper in papers:
            print(f"\nProcessing paper: {paper.title}")
            
            # Check if paper already exists and handle reingestion
            if reingest:
                print(f"Deleting existing chunks for {paper.arxiv_id}...")
                self.vector_store.delete_by_arxiv_id(paper.arxiv_id)
            
            # Download PDF
            pdf_path = self.fetcher.download_pdf(paper.pdf_url, self.papers_dir)
            print(f"Downloaded PDF to {pdf_path}")
            
            # Parse PDF into sections
            sections = self.parser.parse(pdf_path)
            print(f"Parsed into {len(sections)} sections: {list(sections.keys())}")
            
            # Chunk sections
            metadata = {
                "title": paper.title,
                "arxiv_id": paper.arxiv_id,
                "authors": paper.authors,
                "published": paper.published
            }
            chunks = self.chunker.chunk_sections(sections, metadata)
            print(f"Created {len(chunks)} chunks")
            
            # Generate embeddings
            texts = [chunk.text for chunk in chunks]
            embeddings = self.embedder.generate_embeddings(texts)
            print(f"Generated embeddings")
            
            # Store in Qdrant
            self.vector_store.upsert_chunks(chunks, embeddings)
            print(f"Stored chunks in Qdrant")
            
            total_chunks += len(chunks)
            
            # Clean up PDF
            os.remove(pdf_path)
        
        print(f"\n=== Ingestion complete ===")
        print(f"Total chunks ingested: {total_chunks}")
        
        # Print collection info
        collection_info = self.vector_store.get_collection_info()
        print(f"Collection size: {collection_info.points_count} points")
        
        return total_chunks


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run(reingest=False)
