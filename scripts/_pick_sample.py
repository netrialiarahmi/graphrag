#!/usr/bin/env python3
"""Pick sample questions for V4 quick test."""
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

v1_path = os.path.join(ROOT, "output", "qa_benchmark_semantic_govnetic_qa_complete_50_(business).csv")
with open(v1_path, newline="", encoding="utf-8-sig") as f:
    v1 = {r["No"]: r for r in csv.DictReader(f)}

# Pick: all V2-degraded + a few V1-BENAR with refs + a few V1-PARSIAL with refs
sample = set()

# All degraded in V2 (10)
v2_path = os.path.join(ROOT, "output", "qa_benchmark_semantic_v2_govnetic_qa_complete_50_(business).csv")
with open(v2_path, newline="", encoding="utf-8-sig") as f:
    v2 = {r["No"]: r for r in csv.DictReader(f)}
rank = {"BENAR": 0, "PARSIAL": 1, "SALAH": 2}
for no in v1:
    if rank.get(v2[no]["Verdict"], 3) > rank.get(v1[no]["Verdict"], 3):
        sample.add(no)

# All questions with explicit law refs
for no, r in v1.items():
    refs = _extract_doc_references(r["Pertanyaan"])
    if refs:
        sample.add(no)

# Add a few PARSIAL without refs for control
count = 0
for no, r in sorted(v1.items()):
    if no not in sample and r["Verdict"] == "PARSIAL":
        refs = _extract_doc_references(r["Pertanyaan"])
        if not refs:
            sample.add(no)
            count += 1
            if count >= 3:
                break

sample_sorted = sorted(sample)
print(f"Sample: {len(sample_sorted)} questions")
for no in sample_sorted:
    refs = _extract_doc_references(v1[no]["Pertanyaan"])
    print(f"  {no}: V1={v1[no]['Verdict']} V2={v2[no]['Verdict']}  refs={refs if refs else '-'}  Q: {v1[no]['Pertanyaan'][:70]}")

print(f"\nQuestion IDs: {sample_sorted}")
