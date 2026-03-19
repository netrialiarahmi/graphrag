"""Semantic memory & query logging backed by SQLite.
Single-user, local-file persistence for user preferences and analytics."""

import sqlite3
import json
import os
from datetime import datetime


class SemanticMemory:
    def __init__(self, db_path: str = "graphrag_memory.db"):
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                doc_ids TEXT,
                topic TEXT,
                route TEXT,
                latency REAL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS semantic_memory (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self._conn.commit()

    # ── Query logging ─────────────────────────────────────────────────────

    def log_query(self, query: str, doc_ids: list[str] | None = None,
                  topic: str = "", route: str = "", latency: float = 0.0):
        self._conn.execute(
            "INSERT INTO query_log (query, doc_ids, topic, route, latency) VALUES (?, ?, ?, ?, ?)",
            (query, json.dumps(doc_ids or []), topic, route, latency),
        )
        self._conn.commit()

    def get_recent_queries(self, n: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT query, doc_ids, topic, route, latency, created_at "
            "FROM query_log ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [
            {"query": r["query"], "doc_ids": json.loads(r["doc_ids"] or "[]"),
             "topic": r["topic"], "route": r["route"], "latency": r["latency"],
             "time": r["created_at"]}
            for r in rows
        ]

    def get_frequent_topics(self, n: int = 5) -> list[str]:
        rows = self._conn.execute(
            "SELECT topic, COUNT(*) as cnt FROM query_log "
            "WHERE topic != '' GROUP BY topic ORDER BY cnt DESC LIMIT ?", (n,)
        ).fetchall()
        return [r["topic"] for r in rows]

    def get_frequent_docs(self, n: int = 10) -> list[str]:
        """Return the most frequently referenced doc_ids across all queries."""
        rows = self._conn.execute(
            "SELECT doc_ids FROM query_log WHERE doc_ids != '[]'"
        ).fetchall()
        from collections import Counter
        counter: Counter = Counter()
        for r in rows:
            for did in json.loads(r["doc_ids"] or "[]"):
                counter[did] += 1
        return [did for did, _ in counter.most_common(n)]

    def get_user_context_prompt(self) -> str:
        """Build a short context string for the LLM describing user patterns."""
        topics = self.get_frequent_topics(5)
        docs = self.get_frequent_docs(5)
        parts = []
        if topics:
            parts.append(f"Topik yang sering ditanyakan: {', '.join(topics)}.")
        if docs:
            parts.append(f"Dokumen yang sering dirujuk: {', '.join(docs)}.")
        return " ".join(parts)

    # ── Key-value preferences ─────────────────────────────────────────────

    def set_preference(self, key: str, value: str):
        self._conn.execute(
            "INSERT OR REPLACE INTO semantic_memory (key, value, updated_at) "
            "VALUES (?, ?, datetime('now'))", (key, value),
        )
        self._conn.commit()

    def get_preference(self, key: str, default: str = "") -> str:
        row = self._conn.execute(
            "SELECT value FROM semantic_memory WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    # ── Conversation titles for sidebar ───────────────────────────────────

    def save_conversation_title(self, conv_id: str, title: str):
        self.set_preference(f"conv_title:{conv_id}", title)

    def get_all_conversation_titles(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT key, value, updated_at FROM semantic_memory "
            "WHERE key LIKE 'conv_title:%' ORDER BY updated_at DESC"
        ).fetchall()
        return [
            {"id": r["key"].replace("conv_title:", ""), "title": r["value"], "time": r["updated_at"]}
            for r in rows
        ]

    def close(self):
        self._conn.close()
