#!/usr/bin/env python3
"""Diagnose WHY Phase 1 (fetch_by_doc_id) hurt V2 results.

For each V2-degraded question:
1. Does _extract_doc_references find any explicit refs?
2. Which doc_ids were fetched by Phase 1?
3. What content do those fetched chunks contain?
4. Did graph edges add extra docs?
"""
import csv, os, sys, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["GRAPHRAG_STANDALONE"] = "1"

from shared import pinecone_client

# ── _extract_doc_references (same as chatbot) ──
def _extract_doc_references(query):
    results, seen = [], set()
    patterns = [
        (r'(?:Permen\s+PUPR|PERMEN\s+PUPR)\s+(?:No\.?\s*)?(\d+)(?:\s+[Tt]ahun\s+|/)(\d{4})', "PERMENPUPR"),
        (r'(?:Permen\s+PPN|PERMEN\s+PPN)\s+(?:No\.?\s*)?(\d+)(?:\s+[Tt]ahun\s+|/)(\d{4})', "PERMENPPN"),
        (r'(?:Perppu|PERPPU)\s+(?:No\.?\s*)?(\d+)(?:\s+[Tt]ahun\s+|/)(\d{4})', "PERPPU"),
        (r'(?:Pergub|PERGUB)\s+(?:No\.?\s*)?(\d+)(?:\s+[Tt]ahun\s+|/)(\d{4})', "PERGUB"),
        (r'\b(?:UU|Undang-Undang)\s+(?:Nomor\s+|No\.?\s*)?(\d+)(?:\s+[Tt]ahun\s+|/)(\d{4})', "UU"),
        (r'\b(?:PP|Peraturan\s+Pemerintah)\s+(?:Nomor\s+|No\.?\s*)?(\d+)(?:\s+[Tt]ahun\s+|/)(\d{4})', "PP"),
    ]
    for pat, jenis in patterns:
        for m in re.finditer(pat, query, re.IGNORECASE):
            nomor, tahun = m.group(1), m.group(2)
            scope = "PROVINSI" if jenis == "PERGUB" else "NASIONAL"
            doc_id = f"{jenis}-{scope}-{nomor}-{tahun}"
            if doc_id not in seen:
                results.append(doc_id)
                seen.add(doc_id)
    return results


# Load V1/V2 results
v1_path = os.path.join(ROOT, "output", "qa_benchmark_semantic_govnetic_qa_complete_50_(business).csv")
v2_path = os.path.join(ROOT, "output", "qa_benchmark_semantic_v2_govnetic_qa_complete_50_(business).csv")

with open(v1_path, newline="", encoding="utf-8-sig") as f:
    v1 = {r["No"]: r for r in csv.DictReader(f)}
with open(v2_path, newline="", encoding="utf-8-sig") as f:
    v2 = {r["No"]: r for r in csv.DictReader(f)}

rank = {"BENAR": 0, "PARSIAL": 1, "SALAH": 2}

# V2-only degraded (degraded in V2 vs V1)
degraded = []
for no in sorted(v1.keys()):
    v1v, v2v = v1[no]["Verdict"], v2[no]["Verdict"]
    if rank.get(v2v, 3) > rank.get(v1v, 3):
        degraded.append(no)

print(f"=== {len(degraded)} DEGRADED QUESTIONS IN V2 ===\n")

for no in degraded:
    q = v1[no]["Pertanyaan"]
    v1v, v2v = v1[no]["Verdict"], v2[no]["Verdict"]
    v1_docs = v1[no].get("Docs", "")
    v2_docs = v2[no].get("Docs", "")
    refs = _extract_doc_references(q)

    print(f"--- {no}: {v1v} -> {v2v} ---")
    print(f"  Q: {q[:100]}")
    print(f"  Extracted refs: {refs if refs else '(none)'}")
    print(f"  V1 docs: {v1_docs}")
    print(f"  V2 docs: {v2_docs}")

    # Check what V2 has that V1 doesn't
    v1_set = set(d.strip() for d in v1_docs.split(",") if d.strip())
    v2_set = set(d.strip() for d in v2_docs.split(",") if d.strip())
    added = v2_set - v1_set
    removed = v1_set - v2_set
    if added:
        print(f"  ADDED in V2:   {added}")
    if removed:
        print(f"  REMOVED in V2: {removed}")

    # For each extracted ref, check what Pinecone actually has
    if refs:
        for ref in refs:
            try:
                chunks = pinecone_client.fetch_by_doc_id(ref, top_k=5)
                if chunks:
                    print(f"  fetch_by_doc_id('{ref}') -> {len(chunks)} chunks:")
                    for ch in chunks[:3]:
                        content = ch.get("content", "")[:120].replace("\n", " ")
                        print(f"    - [{ch.get('scope','')}] {content}")
                else:
                    print(f"  fetch_by_doc_id('{ref}') -> 0 chunks (EMPTY!)")
            except Exception as e:
                print(f"  fetch_by_doc_id('{ref}') -> ERROR: {e}")
    print()

# Also: how many of the 40 questions trigger Phase 1 at all?
print("\n=== PHASE 1 TRIGGER ANALYSIS (all 40 questions) ===")
triggered = 0
for no in sorted(v1.keys()):
    q = v1[no]["Pertanyaan"]
    refs = _extract_doc_references(q)
    if refs:
        triggered += 1
        print(f"  {no}: {refs}")
print(f"\nTotal: {triggered}/40 questions trigger Phase 1")
