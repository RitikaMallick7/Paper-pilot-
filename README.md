# PaperPilot

An agentic RAG system for research papers with continuous evaluation. PaperPilot ingests arXiv research papers, answers user questions with grounded citations, and continuously evaluates its own answer quality using Ragas metrics.

## Architecture

```
┌─────────────┐
│   Frontend  │ (React + TypeScript + Tailwind)
└──────┬──────┘
       │ HTTP (POST /query)
┌──────▼──────┐
│   Backend   │ (FastAPI)
└──────┬──────┘
       │
┌──────▼──────────────────┐
│  LangGraph Agent         │
│  ┌────────────────────┐  │
│  │ Router → Retriever │  │
│  │ → Answer → Check   │  │
│  └────────────────────┘  │
└──────┬───────────────────┘
       │
┌──────▼──────┐
│   Qdrant    │ (Vector DB - Docker)
└─────────────┘

┌─────────────────────────┐
│ Ingestion Pipeline      │
│ arXiv → PDF → Sections │
│ → Chunks → Embeddings  │
└─────────────────────────┘

┌─────────────────────────┐
│ Evaluation Harness      │
│ Ragas: Faithfulness,    │
│ Relevancy, Precision,  │
│ Recall (tracked over   │
│ time with git commits) │
└─────────────────────────┘
```

## Tech Stack

- **Backend**: Python 3.11+, FastAPI
- **Agent Orchestration**: LangGraph
- **Vector DB**: Qdrant (Docker)
- **Embeddings**: BAAI/bge-small-en-v1.5 (sentence-transformers, local)
- **LLM**: Swappable (Anthropic Claude default, OpenAI supported)
- **Evaluation**: Ragas
- **Frontend**: React + TypeScript + TailwindCSS
- **Infra**: Docker Compose
- **CI**: GitHub Actions (ruff, pytest, tsc)

## Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- Node.js 18+ (for local development)

### Quick Start with Docker

1. Clone the repository:
```bash
git clone <repo-url>
cd paperpilot
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required environment variables:
- `LLM_PROVIDER`: `anthropic` or `openai`
- `ANTHROPIC_API_KEY`: Your Anthropic API key (if using Claude)
- `OPENAI_API_KEY`: Your OpenAI API key (if using GPT)

3. Start all services:
```bash
docker-compose up -d
```

4. Access the services:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Qdrant Dashboard: http://localhost:6333/dashboard

### Local Development

**Backend:**
```bash
# Install dependencies
pip install -r requirements.txt

# Start Qdrant with Docker
docker run -p 6333:6333 qdrant/qdrant:v1.7.0

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run backend
uvicorn backend.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Running Ingestion

The ingestion pipeline fetches papers from arXiv, parses PDFs, chunks by section, generates embeddings, and stores in Qdrant.

### Using Docker
```bash
docker-compose exec backend python ingestion/pipeline.py
```

### Locally
```bash
python ingestion/pipeline.py
```

### Configuration

Edit `.env` to configure ingestion:
- `ARXIV_CATEGORY`: arXiv category (default: `cs.AI`)
- `NUM_PAPERS_TO_INGEST`: Number of papers to fetch (default: `10`)
- `EMBEDDING_MODEL`: Embedding model (default: `BAAI/bge-small-en-v1.5`)

### Idempotent Re-ingestion

To re-ingest papers (deletes existing chunks first):
```bash
python ingestion/pipeline.py  # Add reingest=True flag in code
```

## Using the System

### Via Web Interface

1. Open http://localhost:3000
2. Type a question about research papers
3. View the answer with:
   - Inline citations [1], [2], etc.
   - Grounding confidence badge (Grounded/Medium/Low)
   - Expandable citations panel showing source chunks

### Via API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main contributions of recent AI papers?"}'
```

Response includes:
- `answer`: Generated response with inline citations
- `citations`: List of source chunks with metadata
- `grounding_confidence`: 0-1 score
- `self_check_passed`: Boolean
- `self_check_reasoning`: Explanation of verification
- `query_type`: Classification (direct/retrieval/multi_hop)

## Running Evaluation

The evaluation harness uses Ragas to measure system quality against a gold-standard Q&A set.

### Initial Setup

The gold Q&A set in `eval/gold_qa.py` contains placeholder examples. **TODO**: Manually verify and correct these after running ingestion, or implement the `generate_from_papers()` method to auto-generate questions using an LLM.

### Running Evaluation

```bash
# Docker
docker-compose exec backend python eval/evaluator.py

# Local
python eval/evaluator.py
```

This will:
1. Run the agent on all gold Q&A pairs
2. Compute Ragas metrics (faithfulness, answer relevancy, context precision, context recall)
3. Store results with timestamp and git commit hash in `eval/results/evaluation_results.jsonl`

### Viewing Results

```bash
# Print summary
python eval/plot_results.py

# Generate trend plots (requires 2+ evaluation runs)
python eval/plot_results.py  # Automatically generates plots if enough data
```

### Interpreting Metrics

- **Faithfulness (0-1)**: How well the answer is grounded in retrieved context. Higher is better.
- **Answer Relevancy (0-1)**: How relevant the answer is to the question. Higher is better.
- **Context Precision (0-1)**: How relevant the retrieved context is to the question. Higher is better.
- **Context Recall (0-1)**: How much of the ground truth answer is contained in the retrieved context. Higher is better.

### Tracking Improvement Over Time

Results are stored in JSONL format with git commit hashes. Compare runs to see if changes improve metrics:

```python
from eval.results_store import ResultsStore

store = ResultsStore()
comparison = store.compare_runs(0, 1)  # Compare first and second runs
print(comparison['differences'])
```

## Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_chunker.py -v
```

### Linting
```bash
# Python
ruff check ingestion/ agent/ backend/ eval/

# Frontend (type check)
cd frontend
npx tsc --noEmit
```

## Design Decisions & Trade-offs

### Chunking Strategy
- **Decision**: Character-based chunking with paragraph boundary detection
- **Trade-off**: Token-based chunking would be more accurate for LLM context limits but requires a tokenizer. Character-based is simpler and works well for most cases.

### Hybrid Search
- **Decision**: 70% semantic similarity + 30% keyword matching
- **Trade-off**: BM25 would provide better keyword search but requires additional indexing. Simple keyword scoring is a lightweight alternative.

### Self-Check Threshold
- **Decision**: 0.7 confidence threshold for passing self-check
- **Trade-off**: Lower threshold catches fewer hallucinations but reduces false positives. This can be tuned based on empirical results.

### LLM Provider
- **Decision**: Swappable via environment variable, default to Claude
- **Trade-off**: Claude provides excellent reasoning but requires Anthropic API. OpenAI is supported as fallback.

## Project Structure

```
paperpilot/
├── ingestion/          # arXiv → PDF → Chunks → Embeddings → Qdrant
│   ├── arxiv_fetcher.py
│   ├── pdf_parser.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── pipeline.py
├── agent/              # LangGraph agent nodes
│   ├── router.py
│   ├── retriever.py
│   ├── answer_generator.py
│   ├── self_check.py
│   └── graph.py
├── backend/            # FastAPI application
│   ├── main.py
│   └── config.py
├── eval/               # Ragas evaluation
│   ├── gold_qa.py
│   ├── evaluator.py
│   ├── results_store.py
│   └── plot_results.py
├── frontend/           # React + TypeScript
│   ├── src/
│   │   ├── App.tsx
│   │   └── main.tsx
│   └── package.json
├── tests/              # Pytest tests
│   ├── test_config.py
│   └── test_chunker.py
├── .github/workflows/  # CI/CD
│   └── ci.yml
├── docker-compose.yml
├── Dockerfile.backend
├── requirements.txt
├── .env.example
└── README.md
```

## Troubleshooting

### Qdrant Connection Issues
- Ensure Qdrant is running: `docker ps | grep qdrant`
- Check port: `netstat -an | grep 6333`
- Verify `QDRANT_HOST` and `QDRANT_PORT` in `.env`

### LLM API Errors
- Verify API key is set in `.env`
- Check API key has sufficient credits
- Try switching LLM provider in `.env`

### Frontend Cannot Connect to Backend
- Check backend is running on port 8000
- Verify `VITE_API_URL` environment variable (defaults to `http://localhost:8000`)
- Check CORS settings in FastAPI

### Ingestion Fails
- Ensure arXiv API is accessible
- Check disk space for PDF downloads
- Verify PDF parser dependencies: `pip install pypdf`

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run `pytest` and `ruff check`
6. Submit a pull request
