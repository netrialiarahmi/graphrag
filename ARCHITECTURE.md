# GraphRAG Architecture: Backend + Frontend

This repository now uses a modular architecture with a **FastAPI backend** and a **Streamlit frontend**.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   Streamlit Frontend (app.py)                   │
│                   - Manages chat UI                             │
│                   - Handles visualization display               │
│                   - Calls FastAPI backend via HTTP              │
└────────────────┬──────────────────────────────────────────────┘
                 │
                 │ HTTP POST /query
                 │
┌────────────────▼──────────────────────────────────────────────┐
│            FastAPI Backend (app/api/main.py)                  │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  AgentService (app/services/agent.py)                 │  │
│  │  - Runs LangGraph pipeline                            │  │
│  │  - Returns final answer + narratives                  │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  GraphService (app/services/graph.py)                 │  │
│  │  - Builds D3 visualization payload (JSON)             │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ├─ Neo4j (graph database)                                    │
│  ├─ Pinecone (vector database)                                │
│  └─ LLM via OpenRouter                                        │
└────────────────────────────────────────────────────────────────┘
```

## Running the System

### Prerequisites

Install dependencies:
```bash
pip install -r requirements.txt
```

Set environment variables (create `.env` or export to shell):
```env
OPENROUTER_API_KEY=your_key
HF_AUTH_TOKEN=your_token
HF_ENDPOINT_URL=https://...
HF_MODEL_NAME=Govnetic/Indo-LegalBERT-V3
LLM_MODEL=anthropic/claude-sonnet-4
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=your_index
```

### Start Backend

Terminal 1:
```bash
uvicorn app.api.main:app --reload
```

Backend runs on `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Start Frontend

Terminal 2:
```bash
streamlit run app.py
```

Frontend runs on `http://localhost:8501`

## How It Works

### Query Flow

1. **User enters question** in Streamlit UI
2. **Streamlit calls FastAPI** endpoint with:
   ```json
   {
     "query": "Apa itu dividen interim?",
     "options": {
       "verbose_debug": false,
       "return_logs": true,
       "return_narratives": true
     }
   }
   ```

3. **FastAPI backend runs LangGraph pipeline**:
   - Router node (classifies query as direct/semantic/deep)
   - Lookup/search nodes (retrieves documents)
   - Answer generation node (LLM produces response)

4. **FastAPI returns**:
   ```json
   {
     "answer": "Dividen interim adalah...",
     "narratives": ["Teridentifikasi...", "Saya menemukan..."],
     "primary_doc_ids": ["UU-NASIONAL-8-2022"],
     "d3": {
       "nodes": [...],
       "edges": [...],
       "meta": {...}
     },
     "logs": ["[Context] UU-NASIONAL-8-2022: 25 chunks..."],
     "latency_ms": 3421.5,
     "route": "semantic"
   }
   ```

5. **Streamlit displays**:
   - Answer text
   - Narratives (legal explanations)
   - D3 graph visualization
   - Debug logs (if requested)

## Key Changes from Previous Architecture

| Aspect | Before | After |
|--------|--------|-------|
| **Query Processing** | Streamlit runs agent directly | FastAPI backend handles pipeline |
| **Code Location** | Monolithic `app.py` | Modular `app/` + `utils/` |
| **Checkpointing** | Streamlit manages checkpointer | Backend manages (SQLite/InMemory) |
| **Visualization** | Streamlit renders inline | JSON payload returned; Streamlit renders |
| **Reusability** | Tied to Streamlit | REST API can be used by any client |
| **Scalability** | Single process | Backend can be deployed separately |

## Benefits

✅ **Clean separation** — Frontend (UI) decoupled from backend (logic)
✅ **Reusable** — Backend API can be called by React, Vue, CLI, etc.
✅ **Scalable** — Backend can be deployed to production server
✅ **Testable** — Backend can be tested independently
✅ **Maintainable** — Clear module boundaries and responsibilities

## Troubleshooting

### Streamlit can't reach backend
```
ConnectionError: Could not connect to GraphRAG backend at http://localhost:8000
```
**Fix:** Make sure FastAPI backend is running: `uvicorn app.api.main:app --reload`

### Backend returns 500 error
Check backend logs:
```bash
# In backend terminal, you'll see:
[ERROR] Backend error: ...
```
Fix the error and restart backend.

### Neo4j/Pinecone connection fails
Run health check:
```bash
curl http://localhost:8000/health
```
Returns:
```json
{
  "status": "ok",
  "neo4j": true,
  "pinecone": true
}
```
If `false`, verify credentials in `.env` and that services are running.

## Next Steps

### For Development
- Backend: Modify `app/services/agent.py` or `app/services/graph.py`
- Frontend: Modify visualization code in `app.py`
- Both auto-reload on save (with `--reload` flag)

### For Production
- Deploy backend to production server (e.g., AWS EC2, Google Cloud Run)
- Update Streamlit to point to production backend URL
- Add authentication (API key, OAuth)
- Set up monitoring and logging

### For New Frontends
Instead of Streamlit, build any frontend (React, Vue, CLI) that calls the REST API:

```python
import requests

response = requests.post(
    "http://backend-url:8000/query",
    json={"query": "Bagaimana cara mengajukan gugatan?"}
)
result = response.json()
print(result["answer"])
```

## Files

**Backend:**
- `app/api/main.py` — FastAPI app with endpoints
- `app/services/agent.py` — LangGraph pipeline wrapper
- `app/services/graph.py` — D3 visualization builder
- `app/schemas.py` — Pydantic request/response models

**Frontend:**
- `app.py` — Streamlit UI (now calls backend)

**Utilities:**
- `utils/fastapi_client.py` — HTTP client for calling backend
- `utils/langgraph_agent.py` — LangGraph pipeline (unchanged)
- `utils/` — All other modules (unchanged)

**Documentation:**
- `docs/API.md` — Full API reference
