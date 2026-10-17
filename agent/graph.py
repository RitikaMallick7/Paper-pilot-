"""LangGraph orchestration of all nodes."""
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from agent.router import QueryRouter, RouterState
from agent.retriever import HybridRetriever, RetrieverState
from agent.answer_generator import AnswerGenerator, AnswerState
from agent.self_check import GroundingSelfCheck, SelfCheckState


class AgentState(TypedDict):
    """Combined state for the entire agent graph."""
    query: str
    query_type: str
    retrieved_chunks: list
    answer: str
    citations: list
    grounding_confidence: float
    self_check_passed: bool
    self_check_reasoning: str


class PaperPilotAgent:
    """
    LangGraph agent for RAG with self-checking.
    
    Workflow: Router → Retriever → Answer Generator → Self Check
    """
    
    def __init__(self):
        """Initialize the agent with all nodes."""
        self.router = QueryRouter()
        self.retriever = HybridRetriever()
        self.answer_generator = AnswerGenerator()
        self.self_check = GroundingSelfCheck()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Returns:
            Compiled StateGraph
        """
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("answer_generator", self._answer_node)
        workflow.add_node("self_check", self._self_check_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add edges
        workflow.add_edge("router", "retriever")
        workflow.add_edge("retriever", "answer_generator")
        workflow.add_edge("answer_generator", "self_check")
        workflow.add_edge("self_check", END)
        
        return workflow.compile()
    
    def _router_node(self, state: AgentState) -> AgentState:
        """Router node: classify query type."""
        router_state = RouterState(
            query=state["query"],
            query_type="retrieval",
            reasoning=""
        )
        result = self.router.route(router_state)
        state["query_type"] = result["query_type"]
        return state
    
    def _retriever_node(self, state: AgentState) -> AgentState:
        """Retriever node: fetch relevant chunks."""
        retriever_state = RetrieverState(
            query=state["query"],
            query_type=state["query_type"],
            retrieved_chunks=[],
            retrieval_method=""
        )
        result = self.retriever.retrieve(retriever_state)
        state["retrieved_chunks"] = result["retrieved_chunks"]
        return state
    
    def _answer_node(self, state: AgentState) -> AgentState:
        """Answer generator node: produce answer with citations."""
        answer_state = AnswerState(
            query=state["query"],
            retrieved_chunks=state["retrieved_chunks"],
            answer="",
            citations=[],
            grounding_confidence=0.0
        )
        result = self.answer_generator.generate(answer_state)
        state["answer"] = result["answer"]
        state["citations"] = result["citations"]
        state["grounding_confidence"] = result["grounding_confidence"]
        return state
    
    def _self_check_node(self, state: AgentState) -> AgentState:
        """Self-check node: verify grounding."""
        self_check_state = SelfCheckState(
            query=state["query"],
            answer=state["answer"],
            citations=state["citations"],
            retrieved_chunks=state["retrieved_chunks"],
            grounding_confidence=state["grounding_confidence"],
            self_check_passed=False,
            self_check_reasoning=""
        )
        result = self.self_check.check(self_check_state)
        state["answer"] = result["answer"]
        state["self_check_passed"] = result["self_check_passed"]
        state["self_check_reasoning"] = result["self_check_reasoning"]
        state["grounding_confidence"] = result["grounding_confidence"]
        return state
    
    def query(self, question: str) -> dict:
        """
        Run the agent on a question.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer, citations, and metadata
        """
        initial_state = AgentState(
            query=question,
            query_type="",
            retrieved_chunks=[],
            answer="",
            citations=[],
            grounding_confidence=0.0,
            self_check_passed=False,
            self_check_reasoning=""
        )
        
        final_state = self.graph.invoke(initial_state)
        
        return {
            "answer": final_state["answer"],
            "citations": final_state["citations"],
            "grounding_confidence": final_state["grounding_confidence"],
            "self_check_passed": final_state["self_check_passed"],
            "self_check_reasoning": final_state["self_check_reasoning"],
            "query_type": final_state["query_type"]
        }
