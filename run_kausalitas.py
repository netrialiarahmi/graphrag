#!/usr/bin/env python3
"""
Run Kausalitas analysis — offline script.

Fetches all CITES/HIGHER document pairs from Neo4j, retrieves their content
from Pinecone, and calls the LLM to judge each pair as ENTAILMENT / CONTRADICTION /
NEUTRAL using a comprehensive Indonesian legal framework.

Classification criteria (based on legal doctrine):
  - CONTRADICTION: authority conflicts, contradictory obligations, terminology
    inconsistencies, hierarchy violations (Lex Superior / Specialis / Posterior).
  - ENTAILMENT: delegation/attribution, complementary operationalisation,
    consistent normative alignment.
  - NEUTRAL: different jurisdictional domains, mutually exclusive subject matter,
    no substantive overlap.

Results are written to output/kausalitas/.

Usage:
    python run_kausalitas.py
"""

import csv
import math
import os
import random
import sys
import time

# Mark that we're running standalone (not inside Streamlit)
os.environ["GRAPHRAG_STANDALONE"] = "1"

from dotenv import load_dotenv

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from utils import llm_stance, neo4j_client, pinecone_client

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "kausalitas")

RESULT_COLUMNS = [
    "Dokumen_Sumber",
    "Dokumen_Pembanding",
    "Tipe_Relasi",
    "Kausalitas",
    "Alasan",
]

SUMMARY_COLUMNS = ["Metric", "Value"]

# Cache for fetched document content to avoid redundant VDB calls
_doc_content_cache: dict[str, str] = {}


def _fetch_doc_content(doc_id: str, max_chunks: int = 10) -> str:
    """Fetch concatenated text content for a document from Pinecone.

    Uses a unit random vector with metadata filter {"doc_id": doc_id}
    instead of a zero vector (which produces NaN cosine similarity).
    Results are cached in-memory across calls.
    """
    if doc_id in _doc_content_cache:
        return _doc_content_cache[doc_id]

    index = pinecone_client.get_index()
    stats = pinecone_client.get_index_stats()
    dim = stats.get("dimension", 1024)

    # Create a small unit-ish random vector (not zero — cosine needs non-zero)
    random.seed(42)
    vec = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    vec = [x / norm for x in vec]

    results = index.query(
        vector=vec,
        top_k=max_chunks * 2,
        include_metadata=True,
        filter={"doc_id": doc_id},
    )

    chunks = []
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        content = meta.get("content", "").strip()
        if content:
            chunks.append(content)

    text = "\n\n".join(chunks[:max_chunks])
    _doc_content_cache[doc_id] = text
    return text


def main():
    print("Connecting to Neo4j …")
    try:
        ok = neo4j_client.test_connection()
        if not ok:
            raise RuntimeError("test_connection returned False")
    except Exception as e:
        print(f"[ERROR] Neo4j not reachable: {e}")
        sys.exit(1)

    print("Fetching document pairs from Neo4j …")
    all_pairs = neo4j_client.get_all_document_pairs()
    if not all_pairs:
        print("No document pairs found in Neo4j. Nothing to do.")
        sys.exit(0)

    # Deduplicate: same (src, tgt) may appear with both CITES and HIGHER.
    # Keep both relation types but merge into one analysis call.
    seen_pairs: dict[tuple[str, str], list[str]] = {}
    for pair in all_pairs:
        key = (pair["source_id"], pair["target_id"])
        if key not in seen_pairs:
            seen_pairs[key] = []
        seen_pairs[key].append(pair.get("type", ""))

    unique_pairs = [
        {"source_id": k[0], "target_id": k[1], "types": v}
        for k, v in seen_pairs.items()
    ]

    print(f"Found {len(all_pairs)} edges → {len(unique_pairs)} unique doc pairs.  Starting analysis …\n")

    # Pre-fetch unique documents to avoid redundant VDB calls
    unique_docs: set[str] = set()
    for p in unique_pairs:
        unique_docs.add(p["source_id"])
        unique_docs.add(p["target_id"])
    print(f"Pre-fetching content for {len(unique_docs)} unique documents …")
    for i, doc_id in enumerate(sorted(unique_docs)):
        text = _fetch_doc_content(doc_id)
        status = f"{len(text)} chars" if text else "EMPTY"
        print(f"  [{i+1}/{len(unique_docs)}] {doc_id}: {status}")
    print()

    result_rows: list[dict] = []
    t0 = time.time()

    for idx, pair in enumerate(unique_pairs):
        src_id = pair["source_id"]
        tgt_id = pair["target_id"]
        rel_types = ", ".join(pair["types"])

        pct = (idx + 1) / len(unique_pairs) * 100
        print(f"  [{idx+1}/{len(unique_pairs)}] ({pct:.0f}%) {src_id} → {tgt_id} ({rel_types})", end="", flush=True)

        try:
            text_a = _fetch_doc_content(src_id)
            text_b = _fetch_doc_content(tgt_id)

            if text_a and text_b:
                result = llm_stance.judge_causality(text_a, text_b, src_id, tgt_id)
            else:
                missing = []
                if not text_a:
                    missing.append(src_id)
                if not text_b:
                    missing.append(tgt_id)
                result = {
                    "kausalitas": "NEUTRAL",
                    "alasan": f"Konten tidak tersedia di VDB untuk: {', '.join(missing)}",
                }

            result_rows.append({
                "Dokumen_Sumber": src_id,
                "Dokumen_Pembanding": tgt_id,
                "Tipe_Relasi": rel_types,
                "Kausalitas": result["kausalitas"],
                "Alasan": result["alasan"],
            })
            print(f"  → {result['kausalitas']}")

        except Exception as e:
            result_rows.append({
                "Dokumen_Sumber": src_id,
                "Dokumen_Pembanding": tgt_id,
                "Tipe_Relasi": rel_types,
                "Kausalitas": "Error",
                "Alasan": str(e),
            })
            print(f"  ✗ {e}")

    elapsed = time.time() - t0

    # Summary
    n_contradiction = sum(1 for r in result_rows if r["Kausalitas"] == "CONTRADICTION")
    n_entailment = sum(1 for r in result_rows if r["Kausalitas"] == "ENTAILMENT")
    n_neutral = sum(1 for r in result_rows if r["Kausalitas"] == "NEUTRAL")
    n_error = sum(1 for r in result_rows if r["Kausalitas"] == "Error")

    print(f"\n{'='*60}")
    print(f"  CONTRADICTION : {n_contradiction}")
    print(f"  ENTAILMENT    : {n_entailment}")
    print(f"  NEUTRAL       : {n_neutral}")
    print(f"  Error         : {n_error}")
    print(f"  Total       : {len(result_rows)}")
    print(f"  Elapsed     : {elapsed:.1f}s")
    print(f"{'='*60}")

    # Write CSVs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    result_path = os.path.join(OUTPUT_DIR, "kausalitas_results.csv")
    with open(result_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(result_rows)
    print(f"  → {result_path}")

    summary_path = os.path.join(OUTPUT_DIR, "kausalitas_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for metric, value in [
            ("CONTRADICTION", n_contradiction),
            ("ENTAILMENT", n_entailment),
            ("NEUTRAL", n_neutral),
            ("Error", n_error),
            ("Total", len(result_rows)),
            ("Elapsed_Seconds", f"{elapsed:.1f}"),
        ]:
            writer.writerow({"Metric": str(metric), "Value": str(value)})
    print(f"  → {summary_path}")

    print(f"\nAll done! Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
