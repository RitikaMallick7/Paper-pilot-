"""FastAPI application entry point."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from agent.graph import PaperPilotAgent
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="PaperPilot API", version="0.1.0")

# Initialize the agent
agent = PaperPilotAgent()


class QueryRequest(BaseModel):
    """Request model for the query endpoint."""
    question: str


class Citation(BaseModel):
    """Citation model."""
    index: int
    chunk_id: str
    paper_title: str
    arxiv_id: str
    section: str
    text: str
    score: float





@app.get("/")
def read_root():
    """Root endpoint."""
    return {
        "message": "PaperPilot API",
        "version": "0.1.0",
        "endpoints": {
            "/query": "POST - Ask a question about ingested papers",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
def query_papers(request: QueryRequest):
    """
    Query the RAG system with a question.
    
    Args:
        request: QueryRequest with the question
        
    Returns:
        QueryResponse with answer, citations, and metadata
    """
    try:
        result = agent.query(request.question)
        
        return QueryResponse(
            answer=result["answer"],
            citations=result["citations"],
            grounding_confidence=result["grounding_confidence"],
            self_check_passed=result["self_check_passed"],
            self_check_reasoning=result["self_check_reasoning"],
            query_type=result["query_type"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
