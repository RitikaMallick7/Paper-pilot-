"""Answer generator node with citations."""
from typing import List, Dict, TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from backend.config import config


class AnswerState(TypedDict):
    """State for the answer generator node."""
    query: str
    retrieved_chunks: List[Dict]
    answer: str
    citations: List[Dict]
    grounding_confidence: float


class AnswerGenerator:
    """Generates answers with inline citations."""
    
    def __init__(self):
        """Initialize the answer generator with swappable LLM."""
        self.llm = self._get_llm()
    
    def _get_llm(self):
        """
        Get the LLM based on configuration.
        
        Returns:
            Configured LLM instance
        """
        if config.LLM_PROVIDER == "anthropic":
            if not config.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not set")
            return ChatAnthropic(
                model="claude-3-sonnet-20240229",
                anthropic_api_key=config.ANTHROPIC_API_KEY,
                temperature=0.3
            )
        elif config.LLM_PROVIDER == "openai":
            if not config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set")
            return ChatOpenAI(
                model="gpt-4-turbo-preview",
                openai_api_key=config.OPENAI_API_KEY,
                temperature=0.3
            )
        else:
            raise ValueError(f"Unknown LLM provider: {config.LLM_PROVIDER}")
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """
        Build context string from retrieved chunks.
        
        Args:
            chunks: Retrieved chunks
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_part = f"""
[Source {i}]
Paper: {chunk['paper_title']}
arXiv ID: {chunk['arxiv_id']}
Section: {chunk['section']}
Text: {chunk['text']}
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def generate(self, state: AnswerState) -> AnswerState:
        """
        Generate an answer with inline citations.
        
        Args:
            state: Current state with query and retrieved chunks
            
        Returns:
            Updated state with answer and citations
        """
        query = state["query"]
        chunks = state["retrieved_chunks"]
        
        if not chunks:
            state["answer"] = "I couldn't find relevant information in the ingested papers to answer your question."
            state["citations"] = []
            state["grounding_confidence"] = 0.0
            return state
        
        # Build context
        context = self._build_context(chunks)
        
        # System prompt for citation generation
        system_prompt = """You are a research assistant that answers questions based on provided paper excerpts.
Your answer must:
1. Be accurate and grounded ONLY in the provided context
2. Include inline citations like [1], [2] referencing the source numbers
3. Be concise but comprehensive
4. If the context doesn't contain the answer, state that clearly
5. Never hallucinate information not in the context"""

        user_prompt = f"""Context:
{context}

Question: {query}

Provide a well-structured answer with inline citations."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        answer = response.content
        
        # Extract citations (simple pattern matching)
        # In production, could use more sophisticated parsing
        citations = []
        for i, chunk in enumerate(chunks, 1):
            if f"[{i}]" in answer:
                citations.append({
                    "index": i,
                    "chunk_id": chunk["chunk_id"],
                    "paper_title": chunk["paper_title"],
                    "arxiv_id": chunk["arxiv_id"],
                    "section": chunk["section"],
                    "text": chunk["text"],
                    "score": chunk.get("combined_score", chunk.get("score", 0))
                })
        
        # Initial confidence based on retrieval scores
        if citations:
            avg_score = sum(c["score"] for c in citations) / len(citations)
            state["grounding_confidence"] = min(avg_score * 1.5, 1.0)  # Scale up slightly
        else:
            state["grounding_confidence"] = 0.5  # Medium confidence if no explicit citations
        
        state["answer"] = answer
        state["citations"] = citations
        
        return state
    
    def __call__(self, query: str, chunks: List[Dict]) -> Dict:
        """
        Convenience method to generate an answer.
        
        Args:
            query: User query
            chunks: Retrieved chunks
            
        Returns:
            Dictionary with answer, citations, and confidence
        """
        state = AnswerState(
            query=query,
            retrieved_chunks=chunks,
            answer="",
            citations=[],
            grounding_confidence=0.0
        )
        result = self.generate(state)
        return {
            "answer": result["answer"],
            "citations": result["citations"],
            "grounding_confidence": result["grounding_confidence"]
        }
