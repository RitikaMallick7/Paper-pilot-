"""Module to store embeddings in Qdrant."""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Dict, Optional
from ingestion.chunker import DocumentChunk


class QdrantVectorStore:
    """Manages vector storage in Qdrant."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "paper_chunks",
        embedding_dim: int = 384
    ):
        """
        Initialize the Qdrant client.
        
        Args:
            host: Qdrant host
            port: Qdrant port
            collection_name: Name of the collection
            embedding_dim: Dimension of embeddings
        """
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure the collection exists, create if not."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
    
    def upsert_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """
        Upsert document chunks with their embeddings.
        
        Args:
            chunks: List of DocumentChunk objects
            embeddings: List of embedding vectors corresponding to chunks
        """
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point = PointStruct(
                id=chunk.chunk_id,
                vector=embedding,
                payload={
                    "text": chunk.text,
                    "paper_title": chunk.paper_title,
                    "arxiv_id": chunk.arxiv_id,
                    "authors": chunk.authors,
                    "section": chunk.section,
                    "published": chunk.published
                }
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        arxiv_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query vector
            limit: Number of results to return
            arxiv_id: Optional filter by specific paper
            
        Returns:
            List of search results with payloads
        """
        search_filter = None
        if arxiv_id:
            search_filter = Filter(
                must=[FieldCondition(key="arxiv_id", match=MatchValue(value=arxiv_id))]
            )
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=search_filter,
            with_payload=True
        )
        
        return [
            {
                "score": result.score,
                "text": result.payload["text"],
                "paper_title": result.payload["paper_title"],
                "arxiv_id": result.payload["arxiv_id"],
                "authors": result.payload["authors"],
                "section": result.payload["section"],
                "published": result.payload["published"],
                "chunk_id": result.id
            }
            for result in results
        ]
    
    def delete_by_arxiv_id(self, arxiv_id: str):
        """
        Delete all chunks for a specific paper.
        
        Args:
            arxiv_id: arXiv ID of the paper to delete
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="arxiv_id", match=MatchValue(value=arxiv_id))]
            )
        )
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection."""
        return self.client.get_collection(self.collection_name)
