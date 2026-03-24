#!/usr/bin/env python3
"""Diagnose Phase 1 impact — no Pinecone calls needed."""
import csv, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

# Load all 3 versions
versions = {}
for tag, fname in [
    ("V1", "qa_benchmark_semantic_govnetic_qa_complete_50_(business).csv"),
    ("V2", "qa_benchmark_semantic_v2_govnetic_qa_complete_50_(business).csv"),
    ("V3", "qa_benchmark_semantic_v3_govnetic_qa_complete_50_(business).csv"),
]:
    path = os.path.join(ROOT, "output", fname)
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8-sig") as f:
            versions[tag] = {r["No"]: r for r in csv.DictReader(f)}

v1, v2, v3 = versions["V1"], versions["V2"], versions["V3"]
rank = {"BENAR": 0, "PARSIAL": 1, "SALAH": 2}

# Classify each degraded question
print("=" * 70)
print("CLASSIFICATION OF V2-DEGRADED QUESTIONS")
print("=" * 70)

phase1_triggered = []
phase1_not_triggered = []

for no in sorted(v1.keys()):
    v1v, v2v = v1[no]["Verdict"], v2[no]["Verdict"]
    if rank.get(v2v, 3) <= rank.get(v1v, 3):
        continue  # not degraded
    q = v1[no]["Pertanyaan"]
    refs = _extract_doc_references(q)
    v3v = v3[no]["Verdict"]

    v1_docs = set(d.strip() for d in v1[no].get("Docs", "").split(",") if d.strip())
    v2_docs = set(d.strip() for d in v2[no].get("Docs", "").split(",") if d.strip())
    added = v2_docs - v1_docs
    removed = v1_docs - v2_docs

    bucket = phase1_triggered if refs else phase1_not_triggered
    bucket.append({
        "no": no, "q": q[:90], "refs": refs,
        "v1v": v1v, "v2v": v2v, "v3v": v3v,
        "added": added, "removed": removed,
        "v1_n": len(v1_docs), "v2_n": len(v2_docs),
    })

print(f"\n--- Phase 1 NOT triggered (no explicit law refs in query): {len(phase1_not_triggered)} ---")
for item in phase1_not_triggered:
    print(f"  {item['no']}: {item['v1v']}->{item['v2v']} (V3: {item['v3v']})  docs {item['v1_n']}->{item['v2_n']}")
    print(f"    Q: {item['q']}")
    if item['added']:
        print(f"    ADDED:   {item['added']}")
    if item['removed']:
        print(f"    REMOVED: {item['removed']}")

print(f"\n--- Phase 1 TRIGGERED (explicit law refs in query): {len(phase1_triggered)} ---")
for item in phase1_triggered:
    print(f"  {item['no']}: {item['v1v']}->{item['v2v']} (V3: {item['v3v']})  docs {item['v1_n']}->{item['v2_n']}  refs={item['refs']}")
    print(f"    Q: {item['q']}")
    if item['added']:
        print(f"    ADDED:   {item['added']}")
    if item['removed']:
        print(f"    REMOVED: {item['removed']}")

# Summary
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"Total degraded in V2: {len(phase1_triggered) + len(phase1_not_triggered)}")
print(f"  Phase 1 NOT triggered: {len(phase1_not_triggered)}  (degradation from alpha/top_k/max_docs changes)")
print(f"  Phase 1 TRIGGERED:     {len(phase1_triggered)}  (fetch_by_doc_id was involved)")

# How many recovered in V3 (which removed Phase 1)?
recovered_p1 = sum(1 for x in phase1_triggered if rank.get(x['v3v'],3) <= rank.get(x['v1v'],3))
recovered_nop1 = sum(1 for x in phase1_not_triggered if rank.get(x['v3v'],3) <= rank.get(x['v1v'],3))
still_degraded_p1 = len(phase1_triggered) - recovered_p1
still_degraded_nop1 = len(phase1_not_triggered) - recovered_nop1

print(f"\nRecovery in V3 (Phase 1 removed, kept alpha/top_k/cap changes):")
print(f"  Phase 1 group:   {recovered_p1}/{len(phase1_triggered)} recovered, {still_degraded_p1} still degraded")
print(f"  No-Phase1 group: {recovered_nop1}/{len(phase1_not_triggered)} recovered, {still_degraded_nop1} still degraded")

# Avg doc count comparison
v1_avg = sum(len([d for d in v1[no].get("Docs","").split(",") if d.strip()]) for no in v1) / len(v1)
v2_avg = sum(len([d for d in v2[no].get("Docs","").split(",") if d.strip()]) for no in v2) / len(v2)
v3_avg = sum(len([d for d in v3[no].get("Docs","").split(",") if d.strip()]) for no in v3) / len(v3)
print(f"\nAvg docs/query: V1={v1_avg:.1f}, V2={v2_avg:.1f}, V3={v3_avg:.1f}")
