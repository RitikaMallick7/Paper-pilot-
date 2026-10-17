"""Retriever node for hybrid search."""
from typing import List, Dict, TypedDict
from ingestion.embeddings import EmbeddingGenerator
from ingestion.vector_store import QdrantVectorStore
from backend.config import config
import re


class RetrieverState(TypedDict):
    """State for the retriever node."""
    query: str
    query_type: str
    retrieved_chunks: List[Dict]
    retrieval_method: str


class HybridRetriever:
    """
    Hybrid retriever combining semantic search with keyword matching.
    
    Trade-off: Simple keyword matching (BM25 would be better but requires 
    additional indexing). Using regex-based keyword scoring as a lightweight
    alternative for the hybrid approach.
    """
    
    def __init__(self):
        """Initialize the hybrid retriever."""
        self.embedder = EmbeddingGenerator(model_name=config.EMBEDDING_MODEL)
        self.vector_store = QdrantVectorStore(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            collection_name=config.COLLECTION_NAME,
            embedding_dim=self.embedder.embedding_dim
        )
    
    def _keyword_score(self, chunk: Dict, query: str) -> float:
        """
        Calculate a simple keyword relevance score.
        
        Args:
            chunk: Retrieved chunk with text
            query: User query
            
        Returns:
            Keyword score (0-1)
        """
        query_words = set(re.findall(r'\w+', query.lower()))
        chunk_words = set(re.findall(r'\w+', chunk["text"].lower()))
        
        if not query_words:
            return 0.0
        
        # Count matching words
        matches = len(query_words & chunk_words)
        return matches / len(query_words)
    
    def retrieve(self, state: RetrieverState, top_k: int = 5) -> RetrieverState:
        """
        Retrieve relevant chunks using hybrid search.
        
        Args:
            state: Current state with query
            top_k: Number of chunks to retrieve
            
        Returns:
            Updated state with retrieved chunks
        """
        query = state["query"]
        
        # Generate query embedding
        query_embedding = self.embedder.generate_embedding(query)
        
        # Semantic search
        semantic_results = self.vector_store.search(query_embedding, limit=top_k * 2)
        
        # Combine semantic and keyword scores
        hybrid_results = []
        for result in semantic_results:
            semantic_score = result["score"]
            keyword_score = self._keyword_score(result, query)
            
            # Weighted combination (70% semantic, 30% keyword)
            # Trade-off: This weighting could be tuned based on empirical results
            combined_score = 0.7 * semantic_score + 0.3 * keyword_score
            
            hybrid_results.append({
                **result,
                "combined_score": combined_score
            })
        
        # Sort by combined score and take top-k
        hybrid_results.sort(key=lambda x: x["combined_score"], reverse=True)
        state["retrieved_chunks"] = hybrid_results[:top_k]
        state["retrieval_method"] = "hybrid (semantic + keyword)"
        
        return state
    
    def __call__(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Convenience method to retrieve chunks.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks
        """
        state = RetrieverState(
            query=query,
            query_type="retrieval",
            retrieved_chunks=[],
            retrieval_method=""
        )
        result = self.retrieve(state, top_k)
        return result["retrieved_chunks"]
