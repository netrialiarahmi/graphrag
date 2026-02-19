"""
Generate 6 document availability CSV files from benchmark Excel files.
"""

import os, re, csv, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import openpyxl
from pinecone import Pinecone
from utils import neo4j_client

# ── Gather Neo4j doc_ids ─────────────────────────────────────────────────────
print("Fetching Neo4j documents ...")
neo4j_docs = neo4j_client.get_all_documents()
neo4j_doc_ids = set(d.get("doc_id", "") for d in neo4j_docs)
print(f"  Neo4j: {len(neo4j_doc_ids)} documents")

# ── Gather Pinecone doc_ids ──────────────────────────────────────────────────
print("Fetching Pinecone documents ...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
idx = pc.Index(os.getenv("INDEX_NAME", "lexport-trial"))
stats = idx.describe_index_stats()
dim = stats.get("dimension", 1024)
results = idx.query(vector=[0.0] * dim, top_k=200, include_metadata=True)
pinecone_doc_ids = set()
for m in results.get("matches", []):
    did = m.get("metadata", {}).get("doc_id", "")
    if did:
        pinecone_doc_ids.add(did)
print(f"  Pinecone: {len(pinecone_doc_ids)} unique doc_ids")

# ── Jenis mapping ────────────────────────────────────────────────────────────
JENIS_MAP = {
    "UU": "UU", "PP": "PP", "PERPPU": "PERPPU",
    "PERMEN PUPR": "PERMEN", "PERMEN PPN": "PERMEN",
    "PERMEN PERDAGANGAN": "PERMEN", "PERMEN": "PERMEN",
    "PERMENKES": "PERMENKES", "PERMENDAGRI": "PERMENDAGRI",
    "PERDA": "PERDA", "PERGUB": "PERGUB", "PERWAKO": "PERWAKO",
    "SE": "SE", "SK DIRJEN BK": "KEPDIRJEN", "KEPDIRJEN": "KEPDIRJEN",
    "PERATURAN BPS": "PERMEN", "PERATURAN MENTERI": "PERMEN",
}


def extract_documents(evidence: str) -> list[dict]:
    if not evidence:
        return []
    docs = []
    seen = set()
    patterns = [
        r'(SK\s+Dirjen\s+BK)\s+(\d{4})',
        r'(Peraturan\s+(?:Menteri|BPS|Gubernur|Wali\s+Kota)(?:\s+[A-Za-z]+)*)\s+(\d+/\d{4})',
        r'(Permen\s+[A-Za-z]+)\s+(\d+/\d{4})',
        r'(Permenkes|Permendagri)\s+(\d+/\d{4})',
        r'(PERGUB|Pergub)\s+(\d+/\d{4})',
        r'(Perppu|PERPPU)\s+(\d+/\d{4})',
        r'(Perda\s+[A-Za-z\s]+)\s+(\d+/\d{4})',
        r'(UU|PP)\s+(\d+/\d{4})',
    ]
    for pat in patterns:
        for match in re.finditer(pat, evidence, re.IGNORECASE):
            jenis_raw = match.group(1).strip()
            num_part = match.group(2).strip()
            if "/" in num_part:
                nomor, tahun = num_part.split("/", 1)
            else:
                nomor, tahun = "", num_part
            key = (jenis_raw.upper(), nomor, tahun)
            if key in seen:
                continue
            seen.add(key)
            jenis_upper = jenis_raw.upper().strip()
            jenis_norm = None
            for prefix, mapped in JENIS_MAP.items():
                if jenis_upper.startswith(prefix.upper()):
                    jenis_norm = mapped
                    break
            if not jenis_norm:
                jenis_norm = jenis_upper.split()[0] if jenis_upper else "UNKNOWN"
            candidates = []
            if nomor:
                candidates.append(f"{jenis_norm}-NASIONAL-{nomor}-{tahun}")
                candidates.append(f"{jenis_norm}-PROVINSI-{nomor}-{tahun}")
                candidates.append(f"{jenis_norm}-KOTA-{nomor}-{tahun}")
                if jenis_norm == "KEPDIRJEN":
                    candidates.append(f"KEPDIRJEN-NASIONAL-{nomor}.1-{tahun}")
                if jenis_norm == "PERMENKES":
                    candidates.append(f"PERMENKES-{nomor}-{tahun}")
            docs.append({
                "raw_ref": f"{jenis_raw} {num_part}",
                "jenis": jenis_norm,
                "nomor": nomor,
                "tahun": tahun,
                "candidates": candidates,
            })
    return docs


def find_match(candidates, db_set):
    for c in candidates:
        if c in db_set:
            return c
    return ""


def get_correct_doc_id(doc):
    """Generate the correct doc_id based on jenis: PERGUB->PROVINSI, others->NASIONAL"""
    jenis = doc["jenis"]
    nomor = doc["nomor"]
    tahun = doc["tahun"]
    
    if not nomor:
        return ""
    
    # PERGUB uses PROVINSI scope, all others use NASIONAL
    scope = "PROVINSI" if jenis == "PERGUB" else "NASIONAL"
    return f"{jenis}-{scope}-{nomor}-{tahun}"


def extract_from_excel(path, sheet):
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb[sheet]
    docs_map = {}  # (jenis, nomor, tahun) -> info
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or not row[0]:
            continue
        evidence = str(row[3]) if row[3] else ""
        for doc in extract_documents(evidence):
            key = (doc["jenis"], doc["nomor"], doc["tahun"])
            if key not in docs_map:
                docs_map[key] = doc
    wb.close()
    return list(docs_map.values())


# ── Extract documents from both files ────────────────────────────────────────
print("\nExtracting documents ...")

qa100_docs = extract_from_excel(
    "benchmark/QA 100 (test-all-sector).xlsx", "Gold QA Test Cases"
)
govnetic_docs = extract_from_excel(
    "benchmark/govnetic_qa_complete_50 (business).xlsx", "Govnetic QA Complete 50"
)

print(f"  QA 100: {len(qa100_docs)} unique documents")
print(f"  QA Bisnis: {len(govnetic_docs)} unique documents")

OUT = "benchmark/cek-dokumen-availibility"


def write_doc_list(filename, docs, title):
    path = os.path.join(OUT, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["no", "document_reference", "jenis", "nomor", "tahun", "doc_id"])
        for i, d in enumerate(sorted(docs, key=lambda x: x["raw_ref"]), 1):
            doc_id = get_correct_doc_id(d)
            w.writerow([i, d["raw_ref"], d["jenis"], d["nomor"], d["tahun"], doc_id])
    print(f"  [{len(docs)} rows] {path}")


def write_missing(filename, docs, db_set, db_name):
    missing = [d for d in docs if not find_match(d["candidates"], db_set)]
    path = os.path.join(OUT, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["no", "document_reference", "jenis", "nomor", "tahun", "doc_id"])
        for i, d in enumerate(sorted(missing, key=lambda x: x["raw_ref"]), 1):
            doc_id = get_correct_doc_id(d)
            w.writerow([i, d["raw_ref"], d["jenis"], d["nomor"], d["tahun"], doc_id])
    print(f"  [{len(missing)} missing from {db_name}] {path}")


# ── Generate 6 CSVs ─────────────────────────────────────────────────────────
print("\nGenerating CSVs ...\n")

# 1. List dokumen QA 100
write_doc_list("1_dokumen_qa100.csv", qa100_docs, "QA 100")

# 2. List dokumen QA Bisnis
write_doc_list("2_dokumen_qa_bisnis.csv", govnetic_docs, "QA Bisnis")

# 3. Missing from Neo4j (QA 100)
write_missing("3_missing_neo4j_qa100.csv", qa100_docs, neo4j_doc_ids, "Neo4j")

# 4. Missing from Pinecone/VDB (QA 100)
write_missing("4_missing_vdb_qa100.csv", qa100_docs, pinecone_doc_ids, "Pinecone")

# 5. Missing from Neo4j (QA Bisnis)
write_missing("5_missing_neo4j_qa_bisnis.csv", govnetic_docs, neo4j_doc_ids, "Neo4j")

# 6. Missing from Pinecone/VDB (QA Bisnis)
write_missing("6_missing_vdb_qa_bisnis.csv", govnetic_docs, pinecone_doc_ids, "Pinecone")

# 7. Missing from Neo4j – ALL (QA 100 + QA Bisnis combined)
all_docs = qa100_docs + govnetic_docs
# deduplicate by raw_ref
seen_refs = set()
all_docs_dedup = []
for d in all_docs:
    if d["raw_ref"] not in seen_refs:
        seen_refs.add(d["raw_ref"])
        all_docs_dedup.append(d)
write_missing("7_missing_neo4j_all.csv", all_docs_dedup, neo4j_doc_ids, "Neo4j")

# 8. Missing from Pinecone/VDB – ALL (QA 100 + QA Bisnis combined)
write_missing("8_missing_vdb_all.csv", all_docs_dedup, pinecone_doc_ids, "Pinecone")

print("\nDone!")
