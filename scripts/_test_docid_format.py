#!/usr/bin/env python3
"""Quick test of _extract_doc_references format."""
import re

def _extract_doc_references(query):
    results = []
    seen = set()
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
            nomor = m.group(1)
            tahun = m.group(2)
            scope = "PROVINSI" if jenis == "PERGUB" else "NASIONAL"
            doc_id = f"{jenis}-{scope}-{nomor}-{tahun}"
            if doc_id not in seen:
                results.append(doc_id)
                seen.add(doc_id)
    return results

tests = [
    "Menurut UU 40 tahun 2007 tentang PT",
    "UU 40/2007",
    "UU No. 40 Tahun 2007",
    "Undang-Undang Nomor 40 Tahun 2007 tentang Perseroan Terbatas",
    "Perppu 2 Tahun 2022",
    "PP 29/2021",
    "Permen PPN 7/2023",
]

for q in tests:
    result = _extract_doc_references(q)
    print(f"  {q!r:60s} => {result}")
