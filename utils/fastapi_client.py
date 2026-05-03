"""FastAPI client for calling the GraphRAG backend service.

Provides a lightweight HTTP client that Streamlit and other frontends can use
to query the backend instead of running the pipeline directly.
"""
import requests
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class QueryOptions:
    """Options for query processing."""
    verbose_debug: bool = False
    return_logs: bool = True
    return_narratives: bool = True


@dataclass
class QueryResult:
    """Result from a query."""
    answer: str
    narratives: List[str]
    primary_doc_ids: List[str]
    relationship_context: str
    d3: Dict[str, Any]
    logs: List[str]
    latency_ms: float
    route: str


class FastAPIClient:
    """HTTP client for GraphRAG FastAPI backend."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client.
        
        Args:
            base_url: Base URL of the FastAPI backend (default: localhost:8000)
        """
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
    
    def health_check(self) -> bool:
        """Check if backend is healthy and accessible.
        
        Returns:
            True if backend is reachable and Neo4j/Pinecone are connected.
        """
        try:
            resp = self._session.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
    
    def query(
        self,
        query: str,
        options: Optional[QueryOptions] = None,
        timeout: int = 300,
    ) -> QueryResult:
        """Send a query to the backend and get results.
        
        Args:
            query: Legal question in Indonesian.
            options: QueryOptions for processing parameters.
            timeout: Request timeout in seconds.
        
        Returns:
            QueryResult with answer, narratives, visualization, etc.
        
        Raises:
            requests.exceptions.RequestException: If backend is unreachable.
            ValueError: If backend returns invalid response.
        """
        if options is None:
            options = QueryOptions()
        
        payload = {
            "query": query,
            "options": {
                "verbose_debug": options.verbose_debug,
                "return_logs": options.return_logs,
                "return_narratives": options.return_narratives,
            }
        }
        
        try:
            resp = self._session.post(
                f"{self.base_url}/query",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            
            data = resp.json()
            
            # Validate response structure
            if "answer" not in data:
                raise ValueError("Invalid response: missing 'answer' field")
            
            return QueryResult(
                answer=data.get("answer", ""),
                narratives=data.get("narratives", []),
                primary_doc_ids=data.get("primary_doc_ids", []),
                relationship_context=data.get("relationship_context", ""),
                d3=data.get("d3", {"nodes": [], "edges": [], "meta": {}}),
                logs=data.get("logs", []),
                latency_ms=data.get("latency_ms", 0.0),
                route=data.get("route", "semantic"),
            )
        
        except requests.exceptions.ConnectionError:
            raise requests.exceptions.ConnectionError(
                f"Could not connect to GraphRAG backend at {self.base_url}. "
                "Make sure it's running: uvicorn app.api.main:app --reload"
            )
        except requests.exceptions.Timeout:
            raise requests.exceptions.Timeout(
                f"Backend request timed out after {timeout} seconds"
            )
        except requests.exceptions.HTTPError as e:
            error_detail = "Unknown error"
            try:
                error_detail = e.response.json().get("detail", error_detail)
            except Exception:
                pass
            raise ValueError(f"Backend returned error: {error_detail}")


# Global client instance
_client: Optional[FastAPIClient] = None


def get_client(base_url: str = "http://localhost:8000") -> FastAPIClient:
    """Get or create the global FastAPI client.
    
    Args:
        base_url: Base URL for the backend (if creating new client).
    
    Returns:
        FastAPIClient instance.
    """
    global _client
    if _client is None:
        _client = FastAPIClient(base_url)
    return _client


def reset_client():
    """Reset the global client (useful for testing)."""
    global _client
    _client = None
