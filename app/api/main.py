"""FastAPI application for GraphRAG legal AI backend.

Exposes a single /query endpoint for answering Indonesian legal questions
and returning visualization data in JSON format.
"""
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas import QueryRequest, QueryResponse, D3Payload
from app.services.agent import AgentService
from app.services.graph import build_d3_payload
from utils.helpers import env_bool, setup_file_logger, setup_checkpointer, detect_deployment


# ─────────────────────────────────────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────────────────────────────────────

agent_service: Optional[AgentService] = None
file_logger = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    global agent_service, file_logger
    
    # Startup
    print("🚀 GraphRAG FastAPI backend starting...", flush=True)
    
    # Setup logging
    if env_bool("GRAPHRAG_LOG_TO_FILE", False):
        file_logger = setup_file_logger()
        print("✅ File logging initialized", flush=True)
    
    # Setup checkpointer
    is_deployed = detect_deployment()
    checkpointer = setup_checkpointer(deployed=is_deployed)
    
    # Initialize agent service
    memory_db = os.path.join(os.path.dirname(__file__), "..", "data", "db", "graphrag_memory.db")
    os.makedirs(os.path.dirname(memory_db), exist_ok=True)
    agent_service = AgentService(checkpointer=checkpointer, memory_db=memory_db)
    agent_service.logger = file_logger
    print("✅ Agent service initialized", flush=True)
    
    # Test backend connections
    from utils import neo4j_client, pinecone_client
    neo4j_ok = neo4j_client.test_connection()
    pinecone_ok = pinecone_client.test_connection()
    print(f"📊 Neo4j: {'✅' if neo4j_ok else '❌'}, Pinecone: {'✅' if pinecone_ok else '❌'}", flush=True)
    
    yield
    
    # Shutdown
    print("🛑 GraphRAG backend shutting down...", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GraphRAG Legal AI API",
    description="Backend API for Indonesian legal question answering with graph visualization",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from utils import neo4j_client, pinecone_client
    
    return {
        "status": "ok",
        "neo4j": neo4j_client.test_connection(),
        "pinecone": pinecone_client.test_connection(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Query Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Process a legal question and return answer with visualization data.
    
    Args:
        request: QueryRequest with query and optional parameters.
    
    Returns:
        QueryResponse with answer, narratives, logs, and D3 visualization payload.
    
    Raises:
        HTTPException: If query processing fails or service is unavailable.
    """
    global agent_service
    
    if agent_service is None:
        raise HTTPException(status_code=503, detail="Agent service not initialized")
    
    try:
        # Extract options
        options = request.options or {}
        verbose_debug = options.verbose_debug if hasattr(options, 'verbose_debug') else False
        return_logs = options.return_logs if hasattr(options, 'return_logs') else True
        
        # Run query through agent pipeline
        final_state = agent_service.run_query(
            query=request.query,
            verbose_debug=verbose_debug,
        )
        
        # Build D3 visualization payload
        d3_payload = build_d3_payload(
            context_docs=final_state.get("context_docs", {}),
            relationship_context=final_state.get("relationship_context", ""),
        )
        
        # Prepare response
        response = QueryResponse(
            answer=final_state.get("answer", ""),
            narratives=final_state.get("narratives", []) if options.return_narratives else [],
            primary_doc_ids=final_state.get("primary_doc_ids", []),
            relationship_context=final_state.get("relationship_context", ""),
            d3=D3Payload(**d3_payload),
            logs=final_state.get("logs", []) if return_logs else [],
            latency_ms=final_state.get("latency_ms", 0.0),
            route=final_state.get("route", "semantic"),
        )
        
        return response
        
    except Exception as e:
        import traceback
        error_msg = f"Query processing failed: {str(e)}"
        print(f"❌ {error_msg}\n{traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=error_msg)


# ─────────────────────────────────────────────────────────────────────────────
# Error Handlers
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all exception handler."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=env_bool("DEBUG", False),
    )
