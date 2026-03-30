"""Helpers for detecting conflict-oriented questions and persisting inferred conflict relations."""

import csv
import os
import re

from utils import neo4j_client


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
CONFLICT_OUTPUT_DIR = os.path.join(ROOT_DIR, "output", "conflict")
VISUALIZE_CSV = os.path.join(CONFLICT_OUTPUT_DIR, "visualize_potential_conflict.csv")
SET_CSV = os.path.join(CONFLICT_OUTPUT_DIR, "potential_conflict_set.csv")
VISUALIZE_COLUMNS = ["doc_1", "doc_2", "relation_type", "reasoning"]
SET_COLUMNS = ["doc_1", "doc_2", "relation_type", "question", "reasoning"]


def clear_conflict_output_csv() -> None:
    """Reset ONLY the visualization CSV to header-only for a new question."""
    os.makedirs(CONFLICT_OUTPUT_DIR, exist_ok=True)
    with open(VISUALIZE_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=VISUALIZE_COLUMNS)
        writer.writeheader()


def is_conflict_related_question(query: str) -> bool:
    """Heuristic to decide whether a user query asks about legal conflict/ambiguity."""
    q = (query or "").lower()
    keywords = [
        "konflik",
        "bertentangan",
        "pertentangan",
        "potensi konflik",
        "tumpang tindih",
        "disharmoni",
        "ambiguitas",
        "ambigu",
        "lex specialis",
        "lex superior",
        "lex posterior",
    ]
    return any(k in q for k in keywords)


def _parse_relationship_context_edges(relationship_context: str) -> list[dict]:
    edges: list[dict] = []
    for line in (relationship_context or "").splitlines():
        m = re.match(r"\s*-\s*(.*?)\s*--\[(.*?)\]-->\s*(.*?)\s*$", line)
        if not m:
            continue
        src, rel, tgt = m.group(1), m.group(2), m.group(3)
        if src and tgt:
            edges.append({"doc1": src, "doc2": tgt, "relation": rel or "UNKNOWN"})
    return edges


def _collect_edges_for_conflict_log(primary_doc_ids: list[str], relationship_context: str) -> list[dict]:
    edge_rows: list[dict] = []

    if primary_doc_ids:
        try:
            graph_edges = neo4j_client.get_edges_between(primary_doc_ids).get("edges", [])
            for e in graph_edges:
                src = e.get("source_id", "")
                tgt = e.get("target_id", "")
                rel = e.get("type", "UNKNOWN")
                if src and tgt:
                    edge_rows.append({"doc1": src, "doc2": tgt, "relation": rel})
        except Exception:
            pass

    if not edge_rows:
        edge_rows = _parse_relationship_context_edges(relationship_context)

    if not edge_rows and len(primary_doc_ids) >= 2:
        for i in range(len(primary_doc_ids) - 1):
            edge_rows.append(
                {
                    "doc1": primary_doc_ids[i],
                    "doc2": primary_doc_ids[i + 1],
                    "relation": "UNKNOWN",
                }
            )

    deduped: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for row in edge_rows:
        key = (row["doc1"], row["doc2"], row["relation"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    return deduped


def _normalize_relation_type(conflict_result: dict) -> str:
    """Map inference result to strict relation labels: conflict|entailment."""
    if conflict_result.get("is_conflict"):
        return "conflict"
    label = str(conflict_result.get("label", "")).upper()
    if label == "CONFLICT":
        return "conflict"
    return "entailment"


def _load_existing_compact_rows(path: str) -> list[dict]:
    """Load existing rows and map older headers into compact schema when possible."""
    if not os.path.isfile(path):
        return []

    rows_out: list[dict] = []
    try:
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                doc1 = (
                    r.get("doc_1")
                    or r.get("Doc 1")
                    or r.get("Dokumen_1")
                    or ""
                ).strip()
                doc2 = (
                    r.get("doc_2")
                    or r.get("Doc 2")
                    or r.get("Dokumen_2")
                    or ""
                ).strip()
                rel_raw = (
                    r.get("relation_type")
                    or r.get("Relasi")
                    or r.get("Conflict_Label")
                    or ""
                ).strip().lower()

                if "conflict" in rel_raw or "conflicted" in rel_raw:
                    rel = "conflict"
                else:
                    rel = "entailment"

                if doc1 and doc2:
                    rows_out.append({"doc_1": doc1, "doc_2": doc2, "relation_type": rel})
    except Exception:
        return []

    return rows_out


def _ensure_compact_csv_schema(path: str) -> None:
    """Ensure CSV has compact header; migrate compatible old rows if schema differs."""
    existing_rows = _load_existing_compact_rows(path)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=VISUALIZE_COLUMNS)
        writer.writeheader()
        if existing_rows:
            writer.writerows(existing_rows)


def append_conflict_rows(
    conflict_result: dict,
    primary_doc_ids: list[str],
    relationship_context: str,
    question: str = "",
    reasoning: str = "",
) -> int:
    """Write inferred relations to two outputs:
    - visualize_potential_conflict.csv: reset per question (deduped per run)
    - potential_conflict_set.csv: append full log with question + reasoning
    """
    rows = _collect_edges_for_conflict_log(primary_doc_ids, relationship_context)
    if not rows:
        return 0

    os.makedirs(CONFLICT_OUTPUT_DIR, exist_ok=True)
    relation_type = _normalize_relation_type(conflict_result)
    reason_text = reasoning or conflict_result.get("reason", "") or "-"
    question_text = question or ""

    # Overwrite visualization CSV for current run.
    with open(VISUALIZE_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=VISUALIZE_COLUMNS)
        writer.writeheader()
        seen_out: set[tuple[str, str, str]] = set()
        for edge in rows:
            out_row = {
                "doc_1": edge["doc1"],
                "doc_2": edge["doc2"],
                "relation_type": relation_type,
                "reasoning": reason_text,
            }
            out_key = (out_row["doc_1"], out_row["doc_2"], out_row["relation_type"])
            if out_key in seen_out:
                continue
            seen_out.add(out_key)
            writer.writerow(
                out_row
            )

    # Append to full conflict set log (keep history).
    need_header = not os.path.isfile(SET_CSV)
    with open(SET_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SET_COLUMNS)
        if need_header:
            writer.writeheader()
        for edge in rows:
            writer.writerow(
                {
                    "doc_1": edge["doc1"],
                    "doc_2": edge["doc2"],
                    "relation_type": relation_type,
                    "question": question_text,
                    "reasoning": reason_text,
                }
            )

    return len(seen_out)
