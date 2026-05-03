"""Service layer for the LangGraph agent pipeline.

Wraps the multi-node agentic RAG pipeline and exposes a simple service function
that runs a query and returns the final state (answer, narratives, logs, etc.).
"""
import time
from typing import Dict, Any, List, Optional, cast
from shared.debug_logger import new_trace_id
from utils.memory import SemanticMemory
from utils.helpers import env_bool

# Lazy import to avoid startup issues
_create_agent = None

def _get_create_agent():
    global _create_agent
    if _create_agent is None:
        from utils.langgraph_agent import create_agent
        _create_agent = create_agent
    return _create_agent


class AgentService:
    """Service for running legal question queries through the LangGraph agent."""
    
    def __init__(self, checkpointer=None, memory_db: str = "graphrag_memory.db"):
        """Initialize the agent service.
        
        Args:
            checkpointer: LangGraph checkpointer (SqliteSaver, InMemorySaver, etc.)
                         If None, agent will run without checkpointing.
            memory_db: Path to semantic memory database.
        """
        self.checkpointer = checkpointer
        self.memory = SemanticMemory(memory_db)
        self.logger = None  # Set by FastAPI app if logging is desired
    
    def run_query(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        summary: str = "",
        user_context: str = "",
        verbose_debug: bool = False,
        conv_id: str = None,
    ) -> Dict[str, Any]:
        """Run a query through the agent pipeline.
        
        Args:
            query: User's legal question.
            chat_history: Previous conversation turns.
            summary: Condensed conversation history.
            user_context: Injected semantic memory context.
            verbose_debug: Enable verbose debug logging.
            conv_id: Conversation ID for checkpointing. If None, generates new one.
        
        Returns:
            Dict with keys:
            - answer: final LLM answer
            - narratives: list of user-facing explanations
            - primary_doc_ids: list of key documents
            - context_docs: dict of {doc_id: {chunks, source}}
            - relationship_context: string describing doc relationships
            - logs: list of debug log lines
            - route: pipeline route taken (direct, semantic, deep)
            - latency_ms: processing time in milliseconds
        """
        import uuid
        import sys
        
        _t_start = time.time()
        conv_id = conv_id or str(uuid.uuid4())
        chat_history = chat_history or []
        
        # Retrieve semantic memory context if empty
        if not user_context:
            user_context = self.memory.get_user_context_prompt()
        
        # Build initial state for the agent
        init_state = {
            "query": query,
            "logs": [],
            "narratives": [],
            "primary_doc_ids": [],
            "trace_id": new_trace_id(),
            "verbose_debug": verbose_debug,
            "chat_history": list(chat_history),
            "summary": summary,
            "user_context": user_context,
        }
        init_state = cast(Dict[str, Any], init_state)
        
        # Thread config for checkpointing
        thread_config = {"configurable": {"thread_id": conv_id}}
        thread_config = cast(Dict[str, Any], thread_config)
        
        # Validate checkpointer type
        if self.checkpointer is not None:
            try:
                from langgraph.checkpoint.base import BaseCheckpointSaver
                if not isinstance(self.checkpointer, BaseCheckpointSaver):
                    print(
                        f"[Agent] Invalid checkpointer type: {type(self.checkpointer).__name__}. Fallback to None.",
                        file=sys.stderr,
                    )
                    checkpointer_to_use = None
                else:
                    checkpointer_to_use = self.checkpointer
            except Exception:
                checkpointer_to_use = None
        else:
            checkpointer_to_use = None
        
        # Create agent and run pipeline
        create_agent = _get_create_agent()
        agent = create_agent(checkpointer=checkpointer_to_use)
        
        final_state = {
            "logs": [],
            "narratives": [],
            "primary_doc_ids": [],
            "context_docs": {},
            "answer": "",
            "route": "semantic",
            "relationship_context": "",
        }
        
        try:
            for event in agent.stream(
                cast(Any, init_state),
                config=cast(Any, thread_config),
            ):
                for _node, _update in event.items():
                    final_state.update(_update)
        except Exception as e:
            final_state["logs"].append(f"[Agent Error] {str(e)}")
            final_state["answer"] = f"An error occurred during processing: {str(e)}"
        
        # Update semantic memory with this query
        try:
            self.memory.add_exchange(query, final_state.get("answer", ""))
        except Exception as e:
            if final_state.get("logs"):
                final_state["logs"].append(f"[Memory Error] {str(e)}")
        
        # Calculate latency
        latency_ms = (time.time() - _t_start) * 1000
        final_state["latency_ms"] = latency_ms
        
        # Log the query if logger is set
        if self.logger:
            from utils.helpers import write_log
            write_log(
                self.logger,
                lines=final_state.get("logs", []),
                query=query,
                latency=latency_ms / 1000,
            )
        
        return final_state
