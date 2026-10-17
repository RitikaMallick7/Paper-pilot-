"""Router node to classify query types."""
from typing import Literal, TypedDict
from langchain_core.messages import HumanMessage


class RouterState(TypedDict):
    """State for the router node."""
    query: str
    query_type: Literal["direct", "retrieval", "multi_hop"]
    reasoning: str


class QueryRouter:
    """Routes queries based on their complexity and type."""
    
    def __init__(self):
        """Initialize the router."""
        # Keywords that suggest direct knowledge questions
        self.direct_keywords = [
            "what is", "define", "explain", "who", "when", "where"
        ]
        
        # Keywords that suggest retrieval from papers
        self.retrieval_keywords = [
            "paper", "research", "study", "experiment", "method", "result",
            "according to", "in the paper", "the authors", "findings"
        ]
        
        # Keywords that suggest multi-hop across papers
        self.multi_hop_keywords = [
            "compare", "difference", "between", "versus", "vs",
            "similar", "contrast", "relationship", "both papers"
        ]
    
    def route(self, state: RouterState) -> RouterState:
        """
        Classify the query type.
        
        Args:
            state: Current state with query
            
        Returns:
            Updated state with query_type and reasoning
        """
        query = state["query"].lower()
        
        # Check for multi-hop indicators
        if any(keyword in query for keyword in self.multi_hop_keywords):
            state["query_type"] = "multi_hop"
            state["reasoning"] = "Query contains comparison or cross-paper keywords"
        
        # Check for retrieval indicators
        elif any(keyword in query for keyword in self.retrieval_keywords):
            state["query_type"] = "retrieval"
            state["reasoning"] = "Query asks about specific papers or research"
        
        # Default to direct (will still use retrieval for safety)
        else:
            state["query_type"] = "retrieval"
            state["reasoning"] = "Default to retrieval for grounded answers"
        
        return state
    
    def __call__(self, query: str) -> RouterState:
        """
        Convenience method to route a query.
        
        Args:
            query: The user's question
            
        Returns:
            RouterState with classification
        """
        state = RouterState(query=query, query_type="retrieval", reasoning="")
        return self.route(state)
