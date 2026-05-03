"""Pydantic schemas for FastAPI request/response models."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class QueryOptions(BaseModel):
    """Optional parameters for query processing."""
    verbose_debug: bool = Field(False, description="Enable verbose debug logging")
    return_logs: bool = Field(True, description="Include debug logs in response")
    return_narratives: bool = Field(True, description="Include narrative explanations in response")


class QueryRequest(BaseModel):
    """Request body for /query endpoint."""
    query: str = Field(..., description="User's legal question in Indonesian")
    options: Optional[QueryOptions] = Field(default_factory=QueryOptions, description="Optional processing parameters")


class D3Payload(BaseModel):
    """D3.js visualization payload with nodes and edges."""
    nodes: List[Dict[str, Any]] = Field(default_factory=list, description="Graph nodes with doc_id, labels, etc.")
    edges: List[Dict[str, Any]] = Field(default_factory=list, description="Graph edges with source, target, type, etc.")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the graph (node count, edge count, etc.)")


class QueryResponse(BaseModel):
    """Response body for /query endpoint."""
    answer: str = Field(..., description="Final LLM answer to the user's query")
    narratives: List[str] = Field(default_factory=list, description="User-facing legal explanations from each pipeline stage")
    primary_doc_ids: List[str] = Field(default_factory=list, description="Key document IDs used for answering")
    relationship_context: str = Field(default="", description="Description of relationships between primary documents")
    d3: D3Payload = Field(default_factory=D3Payload, description="D3.js visualization payload (JSON, not HTML)")
    logs: List[str] = Field(default_factory=list, description="Debug logs from the pipeline (if verbose)")
    latency_ms: float = Field(0.0, description="Processing latency in milliseconds")
    route: str = Field("semantic", description="Route taken (direct, semantic, or deep)")
