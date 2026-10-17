"""Self-check node for grounding verification."""
from typing import List, Dict, TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from backend.config import config


class SelfCheckState(TypedDict):
    """State for the self-check node."""
    query: str
    answer: str
    citations: List[Dict]
    retrieved_chunks: List[Dict]
    grounding_confidence: float
    self_check_passed: bool
    self_check_reasoning: str


class GroundingSelfCheck:
    """
    Verifies that claims in the answer are grounded in retrieved chunks.
    
    This is the most important differentiator - it actively checks for hallucinations
    rather than assuming the LLM is always correct.
    """
    
    def __init__(self):
        """Initialize the self-check with swappable LLM."""
        self.llm = self._get_llm()
    
    def _get_llm(self):
        """Get the LLM based on configuration."""
        if config.LLM_PROVIDER == "anthropic":
            if not config.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not set")
            return ChatAnthropic(
                model="claude-3-haiku-20240307",  # Use faster model for checking
                anthropic_api_key=config.ANTHROPIC_API_KEY,
                temperature=0.1
            )
        elif config.LLM_PROVIDER == "openai":
            if not config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set")
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                openai_api_key=config.OPENAI_API_KEY,
                temperature=0.1
            )
        else:
            raise ValueError(f"Unknown LLM provider: {config.LLM_PROVIDER}")
    
    def _extract_claims(self, answer: str) -> List[str]:
        """
        Extract factual claims from the answer.
        
        Args:
            answer: The generated answer
            
        Returns:
            List of claims to verify
        """
        # Simple extraction: split by sentences
        # In production, could use more sophisticated NLP
        import re
        sentences = re.split(r'[.!?]+', answer)
        claims = [s.strip() for s in sentences if s.strip() and len(s) > 20]
        return claims
    
    def check(self, state: SelfCheckState) -> SelfCheckState:
        """
        Verify that the answer is grounded in retrieved chunks.
        
        Args:
            state: Current state with answer and chunks
            
        Returns:
            Updated state with self-check results
        """
        answer = state["answer"]
        chunks = state["retrieved_chunks"]
        
        if not chunks:
            state["self_check_passed"] = False
            state["self_check_reasoning"] = "No chunks retrieved for verification"
            state["grounding_confidence"] = 0.0
            return state
        
        # Build context from chunks
        context = "\n\n".join([
            f"Source {i+1}: {chunk['text']}"
            for i, chunk in enumerate(chunks)
        ])
        
        # Extract claims
        claims = self._extract_claims(answer)
        
        if not claims:
            state["self_check_passed"] = True
            state["self_check_reasoning"] = "No substantive claims to verify"
            return state
        
        # Check each claim
        system_prompt = """You are a fact-checker. Your task is to verify if each claim 
is supported by the provided context. For each claim:
- Respond "SUPPORTED" if the claim is clearly supported by the context
- Respond "NOT_SUPPORTED" if the claim is not supported or contradicted
- Respond "PARTIAL" if partially supported

Be strict - if in doubt, mark as NOT_SUPPORTED."""

        claim_results = []
        for claim in claims:
            user_prompt = f"""Context:
{context}

Claim to verify: {claim}

Is this claim supported by the context? Respond with one word: SUPPORTED, NOT_SUPPORTED, or PARTIAL."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            result = response.content.strip().upper()
            claim_results.append((claim, result))
        
        # Calculate overall grounding
        supported_count = sum(1 for _, result in claim_results if result == "SUPPORTED")
        partial_count = sum(1 for _, result in claim_results if result == "PARTIAL")
        not_supported_count = sum(1 for _, result in claim_results if result == "NOT_SUPPORTED")
        
        total_claims = len(claim_results)
        
        # Calculate confidence score
        # Supported = 1.0, Partial = 0.5, Not supported = 0.0
        confidence = (supported_count + 0.5 * partial_count) / total_claims if total_claims > 0 else 0.0
        
        # Determine if passed (threshold: 0.7)
        passed = confidence >= 0.7
        
        # Build reasoning
        reasoning_parts = [
            f"Checked {total_claims} claims",
            f"{supported_count} fully supported",
            f"{partial_count} partially supported",
            f"{not_supported_count} not supported"
        ]
        
        if not_supported_count > 0:
            unsupported_claims = [claim for claim, result in claim_results if result == "NOT_SUPPORTED"]
            reasoning_parts.append(f"Unsupported claims: {unsupported_claims[:2]}")  # Show first 2
        
        state["self_check_passed"] = passed
        state["self_check_reasoning"] = ". ".join(reasoning_parts)
        state["grounding_confidence"] = confidence
        
        # If failed, modify answer to flag ungrounded portions
        if not passed:
            state["answer"] = self._flag_ungrounded(answer, claim_results)
        
        return state
    
    def _flag_ungrounded(self, answer: str, claim_results: List[tuple]) -> str:
        """
        Add flags to ungrounded portions of the answer.
        
        Args:
            answer: Original answer
            claim_results: List of (claim, verification_result) tuples
            
        Returns:
            Answer with ungrounded portions flagged
        """
        # Simple approach: add warning at the top
        ungrounded_count = sum(1 for _, result in claim_results if result == "NOT_SUPPORTED")
        
        warning = f"\n\n[⚠️ WARNING: {ungrounded_count} claim(s) in this answer could not be verified against the retrieved papers. Please verify independently.]"
        
        return answer + warning
    
    def __call__(self, answer: str, chunks: List[Dict]) -> Dict:
        """
        Convenience method to run self-check.
        
        Args:
            answer: Generated answer
            chunks: Retrieved chunks
            
        Returns:
            Dictionary with self-check results
        """
        state = SelfCheckState(
            query="",
            answer=answer,
            citations=[],
            retrieved_chunks=chunks,
            grounding_confidence=0.0,
            self_check_passed=False,
            self_check_reasoning=""
        )
        result = self.check(state)
        return {
            "answer": result["answer"],
            "self_check_passed": result["self_check_passed"],
            "self_check_reasoning": result["self_check_reasoning"],
            "grounding_confidence": result["grounding_confidence"]
        }
