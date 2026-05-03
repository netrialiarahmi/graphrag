# GraphRAG FastAPI Backend

Modular backend for GraphRAG legal AI, exposing a single REST endpoint for query processing and visualization data.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file (or export to your environment):

```env
OPENROUTER_API_KEY=your_openrouter_key
HF_AUTH_TOKEN=your_huggingface_token
HF_ENDPOINT_URL=https://your-hf-endpoint
HF_MODEL_NAME=Govnetic/Indo-LegalBERT-V3
LLM_MODEL=anthropic/claude-sonnet-4
NEO4J_URI=bolt://your-neo4j-host:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index
DEBUG=false
GRAPHRAG_LOG_TO_FILE=true
```

### 3. Start the Server

```bash
uvicorn app.api.main:app --reload
```

The API will be available at `http://localhost:8000`.

## Endpoints

### Health Check

```bash
GET /health
```

Returns status of backend connections (Neo4j, Pinecone).

### Query Processing

```bash
POST /query
Content-Type: application/json

{
  "query": "Apa itu perusahaan berdasarkan UU 40 tahun 2007?",
  "options": {
    "verbose_debug": false,
    "return_logs": true,
    "return_narratives": true
  }
}
```

**Response:**

```json
{
  "answer": "Perusahaan adalah badan hukum yang didirikan berdasarkan perjanjian...",
  "narratives": [
    "Teridentifikasi rujukan langsung terhadap regulasi UU-NASIONAL-40-2007...",
    "Saya telah menemukan dokumen yang tepat, dan sedang membaca ketentuannya..."
  ],
  "primary_doc_ids": ["UU-NASIONAL-40-2007"],
  "relationship_context": "",
  "d3": {
    "nodes": [
      {
        "id": "UU-NASIONAL-40-2007",
        "label": "UU 40/2007",
        "title": "ID: UU-NASIONAL-40-2007\nJudul: Hukum Perseroan Terbatas\nHierarki: Undang-Undang\nTahun: 2007",
        "size": 35,
        "color": "#1e3a5f",
        "shape": "box",
        "level": 2
      }
    ],
    "edges": [],
    "meta": {
      "node_count": 1,
      "edge_count": 0,
      "primary_doc_ids": ["UU-NASIONAL-40-2007"],
      "relationship_types": []
    }
  },
  "logs": [],
  "latency_ms": 2345.67,
  "route": "direct"
}
```

## API Architecture

```
app/
├── api/
│   └── main.py          # FastAPI app definition and /query endpoint
├── services/
│   ├── agent.py         # Wraps LangGraph pipeline (router → answer)
│   └── graph.py         # Builds D3 visualization payload from context
├── schemas.py           # Pydantic request/response models
└── __init__.py
```

## Key Services

### `AgentService` (`app/services/agent.py`)

Wraps the multi-node LangGraph RAG pipeline. Call `run_query()` to process a legal question:

```python
from app.services.agent import AgentService

service = AgentService()
result = service.run_query(
    query="Bagaimana syarat pendirian perseroan?",
    verbose_debug=False
)
# Returns: {answer, narratives, logs, primary_doc_ids, context_docs, route, ...}
```

### `build_d3_payload()` (`app/services/graph.py`)

Builds JSON visualization data from agent results. Used internally by `/query` endpoint:

```python
from app.services.graph import build_d3_payload

d3_data = build_d3_payload(
    context_docs=result["context_docs"],
    relationship_context=result["relationship_context"]
)
# Returns: {nodes, edges, meta}
```

## Example Client Usage

### cURL

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Apa syarat pendirian PT?",
    "options": {
      "return_logs": false,
      "return_narratives": true
    }
  }' | jq .
```

### Python

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "Bagaimana cara mengubah anggaran dasar PT?",
        "options": {"verbose_debug": True}
    }
)

data = response.json()
print("Answer:", data["answer"])
print("Route:", data["route"])
print("D3 Nodes:", len(data["d3"]["nodes"]))
print("Latency (ms):", data["latency_ms"])
```

### JavaScript / Frontend

The D3 visualization payload can be rendered client-side:

```javascript
fetch('http://localhost:8000/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: 'Apa itu dividen?' })
})
.then(r => r.json())
.then(data => {
  // Render D3 graph
  const nodes = data.d3.nodes;
  const edges = data.d3.edges;
  // ... pass to D3 force simulation
  
  // Display answer
  document.getElementById('answer').textContent = data.answer;
});
```

## Modes and Routing

The agent automatically classifies queries into three processing modes:

1. **direct** — Specific regulation lookup (e.g., "Pasal 5 UU 40/2007")
   - Fast path: regex detection → doc fetch → answer

2. **semantic** — General legal concept (e.g., "syarat pembagian dividen")
   - Method: hybrid BM25 + vector search → LLM rerank → answer
   - Includes sufficiency check; escalates to deep if insufficient

3. **deep** — Complex multi-law analysis (e.g., "konflik UU Cipta Kerja vs UU Ketenagakerjaan")
   - Method: query expansion → multi-term search → graph traversal → reranking → answer

The response includes `route` to indicate which mode was used.

## Logging

Logs are written to `output/logs/app.log` if `GRAPHRAG_LOG_TO_FILE=true`.

Query-level logs are included in the API response (if `options.return_logs=true`).

## Troubleshooting

### Server fails to start
- Check `.env` for missing `OPENROUTER_API_KEY`, Neo4j/Pinecone credentials.
- Ensure PostgreSQL/Neo4j backends are reachable.

### Queries return empty answer
- Check server logs: `tail -f output/logs/app.log`
- Try with `verbose_debug=true` to see reasoning steps.
- Verify Neo4j and Pinecone connections via `/health`.

### Slow responses
- Latency is included in response (`latency_ms`).
- Check if query escalated to `deep` route (more expensive).
- Monitor Neo4j query time in Neo4j browser.

## Development

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

### Code Structure

- **`utils/`** — Reusable modules (Neo4j, Pinecone, LLM, graph viz)
- **`shared/`** — Shared utilities (logging, debug tracing)
- **`app/`** — FastAPI-specific code (services, schemas, API)

Keeps business logic (`utils/`) independent of framework (`app/`).

## Next Steps

- **Auth**: Add API key validation or OAuth for production.
- **Streaming**: Implement SSE or WebSocket for live narrative streaming.
- **Caching**: Cache frequent queries or document graph traversals.
- **Rate Limiting**: Add FastAPI middleware for rate control.
- **Monitoring**: Integrate APM (e.g., Datadog, New Relic).

