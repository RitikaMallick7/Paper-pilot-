"""Module to generate embeddings using sentence-transformers."""
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class EmbeddingGenerator:
    """Generates embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
