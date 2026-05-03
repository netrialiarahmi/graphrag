# GraphRAG — Complete Codebase Reference (API, Services, Utils)

**Comprehensive documentation of all core API endpoints, service methods, utility functions, and database clients used in GraphRAG.** Each function includes full signature, parameters, return values, raises, and usage examples.

## Module Index

- [app/api/main.py](#app-api-main)
- [app/schemas.py](#app-schemas)
- [app/services/agent.py](#app-services-agent)
- [app/services/graph.py](#app-services-graph)
- [utils/helpers.py](#utils-helpers)
- [utils/fastapi_client.py](#utils-fastapi-client)
- [utils/memory.py](#utils-memory)
- [utils/bm25_index.py](#utils-bm25-index)
- [utils/neo4j_client.py](#utils-neo4j-client)
- [utils/pinecone_client.py](#utils-pinecone-client)
- [utils/graph_viz.py](#utils-graph-viz)
- [utils/debug_logger.py](#utils-debug-logger)
- [utils/conflict_logger.py](#utils-conflict-logger)

---

# app/api/main.py

FastAPI backend application exposing a `/query` endpoint for legal Q&A with D3 visualization. Manages app lifecycle with startup/shutdown context.

## FastAPI Application & Lifespan

### `lifespan(app: FastAPI) -> AsyncContextManager`

Async context manager for FastAPI app startup and shutdown lifecycle.

**Signature**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize logger, checkpointer, agent service, test connections
    yield
    # Shutdown: Clean up resources
```

**Startup Behavior**
- Initializes file logger to `output/logs/app.log`
- Creates checkpointer (SQLite or in-memory based on deployment)
- Creates global `AgentService` instance
- Tests Neo4j connection via `neo4j_client.test_connection()`
- Tests Pinecone connection via `pinecone_client.test_connection()`

**Shutdown Behavior**
- Closes checkpointer connections gracefully

**Usage**
```python
# Auto-invoked by FastAPI when creating app
app = FastAPI(lifespan=lifespan)
```

## API Endpoints

### `health_check() -> dict`

Health check endpoint for orchestration and monitoring.

**Signature**
```python
@app.get("/health", tags=["Health"])
async def health_check() -> dict
```

**Returns**
```json
{
  "status": "ok",
  "neo4j": true,
  "pinecone": true
}
```

**HTTP Status**
- `200 OK`

**Example**
```bash
curl http://localhost:8000/health

# Response:
# {"status":"ok","neo4j":true,"pinecone":true}
```

### `query_endpoint(request: QueryRequest) -> QueryResponse`

Main POST endpoint for processing legal questions and returning answers with visualization.

**Signature**
```python
@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_endpoint(request: QueryRequest) -> QueryResponse
```

**Parameters**
- **request** (`QueryRequest`):
  - `query` (str): Legal question in Indonesian
  - `options` (QueryOptions, optional):
    - `verbose_debug` (bool): Enable verbose debug logging to file
    - `return_logs` (bool): Include debug logs in response
    - `return_narratives` (bool): Include narrative explanations

**Returns** (`QueryResponse`)
```python
{
  "answer": str,  # Final LLM answer
  "narratives": list[str],  # Stage-by-stage explanations
  "primary_doc_ids": list[str],  # Key referenced documents
  "relationship_context": str,  # Document relationship descriptions
  "d3": D3Payload,  # Visualization nodes/edges/meta
  "logs": list[str],  # Debug logs (if return_logs=true)
  "latency_ms": float,  # Processing time milliseconds
  "route": str  # Route: "direct", "semantic", or "deep"
}
```

**HTTP Status**
- `200 OK` on success
- `503 Service Unavailable` if AgentService not initialized
- `500 Internal Server Error` on processing failure

**Raises** (Internal)
- Catches all exceptions and returns error message in `answer` field

**Example**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Apa akibat hukum melanggar UU No. 11 Tahun 2020?",
    "options": {
      "verbose_debug": false,
      "return_logs": true,
      "return_narratives": true
    }
  }'
```

**Response Example**
```json
{
  "answer": "Melanggar UU No. 11 Tahun 2020 tentang ITE dapat mengakibatkan...",
  "narratives": [
    "Saya mencari dokumen yang relevan dengan pertanyaan Anda...",
    "Berikut analisis mendalam terhadap topik yang Anda tanyakan..."
  ],
  "primary_doc_ids": ["UU-NASIONAL-11-2020"],
  "relationship_context": "- UU-NASIONAL-11-2020 --[CITES]--> PP-NASIONAL-71-2019",
  "d3": {
    "nodes": [{...}],
    "edges": [{...}],
    "meta": {"node_count": 5, "edge_count": 3}
  },
  "logs": ["[Agent] Starting query processing...", "[Router] Selected route: semantic"],
  "latency_ms": 2450.5,
  "route": "semantic"
}
```

---

# app/schemas.py

Pydantic models for request/response validation in FastAPI. Provides strong typing and automatic OpenAPI schema generation.

### `QueryOptions`

Optional processing parameters for query execution.

**Signature**
```python
class QueryOptions(BaseModel):
    verbose_debug: bool = Field(False, description="Enable verbose debug logging to file")
    return_logs: bool = Field(True, description="Include debug logs in response")
    return_narratives: bool = Field(True, description="Include narrative explanations")
```

**Fields**
- `verbose_debug`: When `True`, writes detailed pipeline trace to `output/logs/debug.log` (JSON lines format)
- `return_logs`: When `True`, includes `logs` array in response JSON
- `return_narratives`: When `True`, includes `narratives` array in response

### `QueryRequest`

Request body for POST `/query` endpoint.

**Signature**
```python
class QueryRequest(BaseModel):
    query: str = Field(..., description="User's legal question in Indonesian")
    options: Optional[QueryOptions] = Field(
        default_factory=QueryOptions,
        description="Optional processing parameters"
    )
```

**Example**
```python
# Sent by client:
{
  "query": "Bagaimana prosedur pembaruan KTP?",
  "options": {"verbose_debug": True, "return_logs": True}
}
```

### `D3Payload`

D3.js visualization data (pure JSON, rendering done client-side).

**Signature**
```python
class D3Payload(BaseModel):
    nodes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Graph node objects"
    )
    edges: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Graph edge objects"
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Graph metadata"
    )
```

**Node Example**
```json
{
  "id": "UU-NASIONAL-11-2020",
  "label": "UU 11/2020",
  "size": 35,
  "color": "#1e3a5f",
  "title": "Undang-Undang ITE\nNomor 11 Tahun 2020",
  "level": 2,
  "shape": "box"
}
```

**Edge Example**
```json
{
  "source": "UU-NASIONAL-11-2020",
  "target": "PP-NASIONAL-71-2019",
  "label": "",
  "color": "#2563eb",
  "type": "CITES",
  "width": 4,
  "dashes": false
}
```

### `QueryResponse`

Full response body for POST `/query`.

**Signature**
```python
class QueryResponse(BaseModel):
    answer: str
    narratives: List[str] = Field(default_factory=list)
    primary_doc_ids: List[str] = Field(default_factory=list)
    relationship_context: str = Field(default="")
    d3: D3Payload = Field(default_factory=D3Payload)
    logs: List[str] = Field(default_factory=list)
    latency_ms: float = Field(0.0)
    route: str = Field("semantic")
```

---

# app/services/agent.py

Service wrapper around the LangGraph agent pipeline. This is the core business logic for query processing.

### `class AgentService`

Orchestrates the LangGraph agent for legal Q&A tasks.

#### `__init__(checkpointer=None, memory_db: str = "graphrag_memory.db")`

Initialize the agent service.

**Signature**
```python
def __init__(self, checkpointer=None, memory_db: str = "graphrag_memory.db")
```

**Parameters**
- **checkpointer** (optional): LangGraph checkpoint saver
  - Type: `langgraph.checkpoint.base.BaseCheckpointSaver`
  - Examples: `SqliteSaver(conn)`, `InMemorySaver()`
  - If `None`, agent runs without state persistence
- **memory_db** (str): Path to semantic memory SQLite database
  - Default: `"graphrag_memory.db"`
  - Recommended: `"data/db/graphrag_memory.db"`

**Example**
```python
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

conn = sqlite3.connect("data/db/checkpointer.db")
checkpointer = SqliteSaver(conn)
svc = AgentService(
    checkpointer=checkpointer,
    memory_db="data/db/graphrag_memory.db"
)
```

#### `run_query(query, chat_history=None, summary="", user_context="", verbose_debug=False, conv_id=None) -> Dict`

Execute a legal question through the agent pipeline.

**Signature**
```python
def run_query(
    self,
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    summary: str = "",
    user_context: str = "",
    verbose_debug: bool = False,
    conv_id: str = None,
) -> Dict[str, Any]
```

**Parameters**
- **query** (str): User's legal question (required)
- **chat_history** (list, optional): Previous conversation turns
  - Format: `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`
  - If `None`, starts fresh conversation
- **summary** (str): Condensed history for context window
- **user_context** (str): Injected user context/preferences
  - If empty, retrieves from `SemanticMemory`
- **verbose_debug** (bool): Enable verbose logging to `output/logs/debug.log`
- **conv_id** (str, optional): Conversation ID for checkpointing
  - If `None`, generates new UUID

**Returns** (Dict)
```python
{
  "answer": str,  # Final LLM-generated answer
  "narratives": list[str],  # Stage explanations
  "primary_doc_ids": list[str],  # Key document IDs
  "context_docs": dict,  # {doc_id: {chunks: [...], source: "..."}}
  "relationship_context": str,  # Relationship descriptions
  "logs": list[str],  # Debug logs
  "route": str,  # "direct", "semantic", or "deep"
  "latency_ms": float  # Processing time
}
```

**Raises** (Internal)
- Catches exceptions; returns error message in `answer` with `route="error"`

**Example**
```python
result = svc.run_query(
    query="Apa itu UU No. 11 Tahun 2020?",
    verbose_debug=True,
    conv_id="user-session-123"
)

print(result['answer'])
print(f"Latency: {result['latency_ms']}ms")
print(f"Route: {result['route']}")
print(f"Documents: {result['primary_doc_ids']}")
```

**Design Notes**
- Uses lazy import for LangGraph: `from utils.langgraph_agent import create_agent`
- Lazy import avoids startup failures in minimal environments
- Automatically logs query to `SemanticMemory` for user context learning
- If `file_logger` is set, writes detailed logs to file

---

# app/services/graph.py

Service for building D3.js visualization payloads from agent context documents.

### `build_d3_payload(context_docs: Dict[str, Any], relationship_context: str = "", doc_metadata: Dict[str, Any] = None) -> Dict`

Build D3-compatible graph JSON from context documents and relationships.

**Signature**
```python
def build_d3_payload(
    context_docs: Dict[str, Any],
    relationship_context: str = "",
    doc_metadata: Dict[str, Any] = None,
) -> Dict[str, Any]
```

**Parameters**
- **context_docs** (dict): Document context mapping
  - Format: `{doc_id: {chunks: [str, ...], source: "Neo4j"|"Pinecone"|...}, ...}`
  - Example: `{"UU-NASIONAL-11-2020": {chunks: ["Pasal 1...", "Pasal 2..."], source: "Neo4j"}}`
- **relationship_context** (str, optional): Relationship descriptions
  - Format: Multi-line, each line: `- SOURCE_ID --[RELATION_TYPE]--> TARGET_ID`
  - Example: `"- UU-NASIONAL-11-2020 --[CITES]--> PP-NASIONAL-71-2019\n- PP-NASIONAL-71-2019 --[IMPLEMENTS]--> UU-NASIONAL-11-2020"`
- **doc_metadata** (dict, optional): Document properties for enrichment
  - Format: `{doc_id: {jenis: "UU", judul: "...", tahun: 2020, ...}, ...}`

**Returns** (Dict)
```python
{
  "nodes": list[dict],  # D3 node objects
  "edges": list[dict],  # D3 edge objects
  "meta": {
    "node_count": int,
    "edge_count": int,
    "primary_doc_ids": list[str],
    "relationship_types": set
  }
}
```

**Example**
```python
payload = build_d3_payload(
    context_docs={
        "UU-NASIONAL-11-2020": {
            "chunks": ["Pasal 1 berbunyi...", "Pasal 2 berbunyi..."],
            "source": "Neo4j"
        },
        "PP-NASIONAL-71-2019": {
            "chunks": ["Pasal 5 berbunyi..."],
            "source": "Pinecone"
        }
    },
    relationship_context="- UU-NASIONAL-11-2020 --[CITES]--> PP-NASIONAL-71-2019",
    doc_metadata={
        "UU-NASIONAL-11-2020": {"jenis": "UU", "judul": "ITE", "tahun": 2020},
        "PP-NASIONAL-71-2019": {"jenis": "PP", "judul": "Impor", "tahun": 2019}
    }
)

import json
print(json.dumps(payload, indent=2))
```

### `build_d3_html(d3_payload, selected_doc_id=None, label_mode="Doc ID", charge=-320, link_distance=90) -> str`

Server-side convenience to render D3 HTML. Typically frontends prefer raw JSON.

**Signature**
```python
def build_d3_html(
    d3_payload: Dict[str, Any],
    selected_doc_id: str = None,
    label_mode: str = "Doc ID",
    charge: int = -320,
    link_distance: int = 90,
) -> str
```

**Parameters**
- **d3_payload**: Output from `build_d3_payload()`
- **selected_doc_id**: Document ID to highlight (yellow border)
- **label_mode**: Display mode: "Doc ID", "Short", or other
- **charge**: D3 force repulsion (lower = more spread)
- **link_distance**: D3 link length in pixels

**Returns**
- **str**: HTML/JavaScript embedding D3 force simulation

**Note**: For Streamlit integration, return raw `d3_payload` JSON and use JavaScript rendering on client.

---

# utils/helpers.py

Reusable helper functions for environment config, logging, and checkpointer setup.

### `env_bool(name: str, default: bool = False) -> bool`

Parse environment variable as boolean.

**Signature**
```python
def env_bool(name: str, default: bool = False) -> bool
```

**Parameters**
- **name** (str): Environment variable name (e.g., "DEBUG", "VERBOSE_DEBUG")
- **default** (bool): Default if variable not set

**Returns**
- **bool**: `True` if value is "1", "true", "yes", or "on" (case-insensitive)

**Example**
```python
debug = env_bool("GRAPHRAG_VERBOSE_DEBUG", False)
use_gpu = env_bool("CUDA_AVAILABLE", False)
```

### `setup_file_logger(log_dir: str = None, log_filename: str = "app.log") -> logging.Logger`

Create a file logger for the application.

**Signature**
```python
def setup_file_logger(
    log_dir: str = None,
    log_filename: str = "app.log"
) -> logging.Logger
```

**Parameters**
- **log_dir** (str, optional): Directory for log files (default: `"output/logs"`)
- **log_filename** (str): Log file name (default: `"app.log"`)

**Returns**
- **logging.Logger**: Configured logger (singleton per name)

**Example**
```python
logger = setup_file_logger(log_filename="backend.log")
logger.info("Application started")
logger.debug("Debug message")
```

### `write_log(logger, lines: list[str] | None, query: str = "", latency: float = 0.0)`

Append agent logs to file logger.

**Signature**
```python
def write_log(
    logger: logging.Logger,
    lines: list[str] = None,
    query: str = "",
    latency: float = 0.0
)
```

**Parameters**
- **logger**: Logger from `setup_file_logger()`
- **lines**: List of debug log lines
- **query**: User query (context)
- **latency**: Response time in seconds

**Example**
```python
logs = ["[Router] Selected semantic...", "[Search] Found 5 documents"]
write_log(logger, logs, query="Apa itu pajak?", latency=2.5)
# Writes: "query=Apa itu pajak? | latency=2.5s"
#         "  [Router] Selected semantic..."
#         "  [Search] Found 5 documents"
```

### `setup_checkpointer(deployed: bool = False)`

Initialize a LangGraph checkpointer.

**Signature**
```python
def setup_checkpointer(deployed: bool = False)
```

**Parameters**
- **deployed** (bool): Environment flag
  - If `True`: Returns `InMemorySaver` (ephemeral)
  - If `False`: Returns `SqliteSaver` at `data/db/checkpointer.db` (persistent)

**Returns**
- BaseCheckpointSaver or `None` on failure

**Example**
```python
is_prod = detect_deployment()
checkpointer = setup_checkpointer(deployed=is_prod)
if checkpointer is None:
    print("Warning: Checkpointer not initialized")
```

### `detect_deployment() -> bool`

Detect deployed runtime environment.

**Signature**
```python
def detect_deployment() -> bool
```

**Returns**
- **bool**: `True` if running on Streamlit Cloud, Docker, or production

**Detection**
- Environment variable `ENVIRONMENT=production`
- File exists: `/.dockerenv`
- Environment variable `STREAMLIT_SERVER_RUNDIR` exists

**Example**
```python
if detect_deployment():
    print("Running in production")
else:
    print("Running locally")
```

---

# utils/fastapi_client.py

HTTP client library for calling the GraphRAG FastAPI backend from Streamlit or other frontends.

### `class QueryOptions`

Dataclass for query processing options.

**Signature**
```python
@dataclass
class QueryOptions:
    verbose_debug: bool = False
    return_logs: bool = True
    return_narratives: bool = True
```

### `class QueryResult`

Dataclass for parsed query response.

**Signature**
```python
@dataclass
class QueryResult:
    answer: str
    narratives: List[str]
    primary_doc_ids: List[str]
    relationship_context: str
    d3: Dict[str, Any]
    logs: List[str]
    latency_ms: float
    route: str
```

### `class FastAPIClient`

HTTP client for backend.

#### `__init__(base_url: str = "http://localhost:8000")`

Initialize the client.

**Parameters**
- **base_url** (str): Backend base URL (default: `"http://localhost:8000"`)

**Example**
```python
client = FastAPIClient(base_url="http://localhost:8000")
```

#### `health_check() -> bool`

Check if backend is healthy.

**Signature**
```python
def health_check(self) -> bool
```

**Returns**
- **bool**: `True` if backend responds 200 OK

**Example**
```python
if client.health_check():
    print("Backend is running")
```

#### `query(query: str, options: QueryOptions = None, timeout: int = 300) -> QueryResult`

Send query to backend and get results.

**Signature**
```python
def query(
    self,
    query: str,
    options: Optional[QueryOptions] = None,
    timeout: int = 300,
) -> QueryResult
```

**Parameters**
- **query** (str): Legal question in Indonesian
- **options** (QueryOptions): Processing options (default: `QueryOptions()`)
- **timeout** (int): Request timeout seconds (default: 300)

**Returns**
- **QueryResult**: Parsed response

**Raises**
- **requests.exceptions.ConnectionError**: Backend unreachable
  - Helpful message suggests: `uvicorn app.api.main:app --reload`
- **requests.exceptions.Timeout**: Exceeded timeout
- **ValueError**: Invalid response structure

**Example**
```python
client = FastAPIClient()
try:
    result = client.query(
        "Apa itu UU No. 11 Tahun 2020?",
        options=QueryOptions(verbose_debug=True),
        timeout=300
    )
    print(result.answer)
    print(f"Latency: {result.latency_ms}ms")
except Exception as e:
    print(f"Error: {e}")
```

### `get_client(base_url: str = "http://localhost:8000") -> FastAPIClient`

Get or create global FastAPI client (singleton).

**Signature**
```python
def get_client(base_url: str = "http://localhost:8000") -> FastAPIClient
```

**Returns**
- **FastAPIClient**: Global singleton

**Example**
```python
from utils.fastapi_client import get_client
client = get_client()
result = client.query("...")
```

### `reset_client()`

Reset global client (testing).

**Signature**
```python
def reset_client()
```

---

# utils/memory.py

SQLite-backed semantic memory and query logging for single-user persistence.

### `class SemanticMemory`

Semantic memory and query analytics backed by SQLite.

#### `__init__(db_path: str = "graphrag_memory.db")`

Initialize memory store.

**Parameters**
- **db_path** (str): SQLite file path (default: `"graphrag_memory.db"`, recommended: `"data/db/graphrag_memory.db"`)

**Tables**
- `query_log`: Records queries with doc_ids, topic, route, latency
- `semantic_memory`: Key-value preferences store

**Example**
```python
mem = SemanticMemory("data/db/graphrag_memory.db")
```

#### `log_query(query: str, doc_ids: list[str] | None = None, topic: str = "", route: str = "", latency: float = 0.0)`

Record a user query and metadata.

**Example**
```python
mem.log_query(
    query="Apa itu pajak penghasilan?",
    doc_ids=["UU-NASIONAL-36-2008"],
    topic="pajak",
    route="semantic",
    latency=2.5
)
```

#### `get_recent_queries(n: int = 20) -> list[dict]`

Get recent query logs.

**Returns**
- **list[dict]**: Last `n` queries

**Example**
```python
recent = mem.get_recent_queries(10)
for q in recent:
    print(f"{q['query']} ({q['latency']}s, route={q['route']})")
```

#### `get_frequent_topics(n: int = 5) -> list[str]`

Get top topics by frequency.

**Returns**
- **list[str]**: Top `n` topic strings

#### `get_frequent_docs(n: int = 10) -> list[str]`

Get most referenced documents.

**Returns**
- **list[str]**: Top `n` doc IDs

#### `get_user_context_prompt() -> str`

Build context summary for LLM.

**Returns**
- **str**: Summary like `"Topik yang sering ditanyakan: pajak, ketenagakerjaan. Dokumen yang sering dirujuk: UU-36-2008, PP-71-2019"`

**Example**
```python
context = mem.get_user_context_prompt()
# Use in LLM system prompt
```

#### `set_preference(key: str, value: str)` & `get_preference(key: str, default: str = "") -> str`

Store/retrieve user preferences.

**Example**
```python
mem.set_preference("favorite_docs", "UU-11-2020,PP-71-2019")
fav = mem.get_preference("favorite_docs")
```

#### `save_conversation_title(conv_id: str, title: str)` & `get_all_conversation_titles() -> list[dict]`

Save/retrieve conversation history titles.

**Example**
```python
mem.save_conversation_title("conv-123", "Pertanyaan tentang pajak")
convs = mem.get_all_conversation_titles()
```

---

# utils/bm25_index.py

In-memory BM25 keyword search index over Pinecone corpus with hybrid dense+BM25 fusion.

### `bm25_search(query: str, top_k: int = 50) -> list[dict]`

Run BM25 keyword search.

**Signature**
```python
def bm25_search(query: str, top_k: int = 50) -> list[dict]
```

**Returns**
- **list[dict]**: Results with `id`, `doc_id`, `article_id`, `content`, `scope`, `bm25_score`

**Example**
```python
hits = bm25_search("pajak penghasilan", top_k=10)
for h in hits:
    print(f"{h['doc_id']}: {h['bm25_score']:.2f} - {h['content'][:100]}")
```

### `refresh_cache()`

Force re-download and rebuild BM25 index.

**Example**
```python
refresh_cache()  # Call after new documents added to Pinecone
```

### `hybrid_search(query: str, query_embedding: list[float], top_k: int = 20, alpha: float = 0.5) -> list[dict]`

Hybrid search with RRF (Reciprocal Rank Fusion).

**Signature**
```python
def hybrid_search(
    query: str,
    query_embedding: list[float],
    top_k: int = 20,
    alpha: float = 0.5,
) -> list[dict]
```

**Parameters**
- **alpha**: Weight for dense vs BM25 (0.5=balanced, >0.5=prefer semantic, <0.5=prefer keyword)

**Example**
```python
embedding = embed_query("pajak penghasilan")
hits = hybrid_search("pajak penghasilan", embedding, top_k=10, alpha=0.6)
```

---

# utils/neo4j_client.py

Neo4j document graph connector.

### `test_connection() -> bool`

Test Neo4j connection health.

### `get_all_documents() -> list[dict]`

Fetch all Document nodes.

**Returns**
- **list[dict]**: Documents with `doc_id`, `judul`, `jenis`, `tahun`, `nomor`, `pembentuk`

### `get_document_detail(doc_id: str) -> dict`

Fetch single document with children (Pasal, Ayat, Diktum, Lampiran).

**Returns**
- **dict**: `{document, pasals, ayats, diktums, lampirans}`

### `get_document_subgraph(doc_ids: list[str]) -> dict`

Fetch subgraph for documents.

**Returns**
- **dict**: `{nodes: [...], edges: [...]}`

### `get_citing_documents(doc_id: str, hops: int = 2) -> dict`

Fetch k-hop neighbors via CITES/HIGHER.

### `get_edges_between(doc_ids: list[str]) -> dict`

Fetch edges connecting documents in list.

### `get_related_documents(doc_id: str, limit: int = 3) -> list[dict]`

Fetch directly related documents.

### `search_lampiran_content(keywords: list[str], max_docs: int = 5) -> list[str]`

Search attachment content by keyword.

---

# utils/pinecone_client.py

Pinecone dense retrieval connector.

### `test_connection() -> bool`

Test Pinecone connection.

### `get_index_stats() -> dict`

Get index statistics (dimension, total_vectors, namespaces).

### `semantic_search(query_embedding: list[float], top_k: int = 10, scope_filter: str = None) -> list[dict]`

Search with embedding vector.

**Returns**
- **list[dict]**: Results with `id`, `doc_id`, `article_id`, `content`, `scope`, `score`

### `fetch_by_doc_id(doc_id: str, top_k: int = 100) -> list[dict]`

Fetch all vectors for document.

### `fetch_by_ids(ids: list[str]) -> list[dict]`

Fetch specific vectors by ID.

---

# utils/graph_viz.py

Graph visualization utilities (streamlit-agraph and D3).

### Constants

```python
NODE_COLORS = {"Document": "#1e3a5f", "Pasal": "#d97706", ...}
HIERARCHY_LEVEL_NAMES = {0: "UUD 1945", 1: "Ketetapan MPR", ...}
```

### `_get_hierarchy_level(doc_id: str, jenis: str = "") -> int`

Get hierarchy level (0-14).

### `_get_short_label(doc_id: str) -> str`

Generate display label (e.g., "UU 11/2020").

### `render_graph(nodes, edges, stance_map=None, height=500, physics=True)`

Render non-hierarchical graph.

### `render_document_graph(doc_nodes, doc_edges, stance_map=None, height=600)`

Render hierarchical document graph.

### `build_d3_html(...) -> str`

Build D3 HTML/JS string.

### `merge_graph_payload(existing, new) -> dict`

Merge two payloads without duplicates.

---

# utils/debug_logger.py

Structured JSON-lines debug logging.

### `new_trace_id() -> str`

Generate unique trace ID.

**Example**
```python
trace = new_trace_id()  # "a1b2c3d4-e5f6-47a8-9b0c-d1e2f3a4b5c6"
```

### `is_verbose_debug_enabled() -> bool`

Check if `GRAPHRAG_VERBOSE_DEBUG=true`.

### `log_event(trace_id, route, stage, event, message, payload=None)`

Log structured event to `output/logs/debug.log`.

### `log_verbose_event(route, stage, event, message, payload=None, trace_id="")`

Conditional verbose logging.

---

# utils/conflict_logger.py

Conflict detection and CSV logging.

### `is_conflict_related_question(query: str) -> bool`

Detect conflict-oriented queries.

**Keywords**: "konflik", "pertentangan", "ambiguitas", "lex specialis", etc.

### `clear_conflict_output_csv()`

Reset visualization CSV for new question.

**File**: `output/conflict/visualize_potential_conflict.csv`

### `append_conflict_rows(conflict_result, primary_doc_ids, relationship_context, question="", reasoning="") -> int`

Write inferred relations to two CSVs.

**Files**
- `output/conflict/visualize_potential_conflict.csv` (reset per question)
- `output/conflict/potential_conflict_set.csv` (append all history)

---

## Common Usage Patterns

### Query via FastAPI Client
```python
from utils.fastapi_client import get_client, QueryOptions

client = get_client()
result = client.query(
    "Apa itu UU No. 11 Tahun 2020?",
    options=QueryOptions(verbose_debug=True),
    timeout=300
)
print(result.answer)
```

### Direct Service Query
```python
from app.services.agent import AgentService
from utils.helpers import setup_checkpointer

checkpointer = setup_checkpointer()
svc = AgentService(checkpointer=checkpointer)
result = svc.run_query("Apa akibat hukum melanggar UU 11/2020?")
print(result['answer'])
```

### Hybrid Search
```python
from utils.bm25_index import hybrid_search
from utils.pinecone_client import semantic_search

embedding = embed_query("pajak penghasilan")
hits = hybrid_search("pajak penghasilan", embedding, top_k=10)
```

---

**Last Updated**: May 3, 2026  
**Version**: 1.1
