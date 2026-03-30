"""Check UU 11/2020 for dividen content + cross-check with PERPPU 2/2022."""
import os, sys
os.environ['GRAPHRAG_STANDALONE'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv; load_dotenv()
from utils import neo4j_client, pinecone_client, llm_stance

driver = neo4j_client.get_driver()
with driver.session() as s:
    # 1. UU 11/2020 pasals with dividen
    r = s.run(
        'MATCH (d:Document {doc_id: "UU-NASIONAL-11-2020"})-[:HAS_PASAL]->(p:Pasal) '
        'WHERE toLower(p.content) CONTAINS "dividen" '
        'RETURN p.name AS name, p.content AS content ORDER BY p.name'
    )
    print("=== UU-NASIONAL-11-2020 pasals with 'dividen' ===")
    for rec in r:
        print(f"  {rec['name']}: {rec['content'][:400]}")
        print()

    # 2. UU 11/2020 ayats with dividen
    r2 = s.run(
        'MATCH (d:Document {doc_id: "UU-NASIONAL-11-2020"})-[:HAS_PASAL]->(p:Pasal)-[:HAS_AYAT]->(a:Ayat) '
        'WHERE toLower(a.content) CONTAINS "dividen" '
        'RETURN p.name AS pasal, a.name AS ayat, a.content AS content ORDER BY p.name, a.name'
    )
    print("=== UU-NASIONAL-11-2020 ayats with 'dividen' ===")
    for rec in r2:
        print(f"  {rec['pasal']}/{rec['ayat']}: {rec['content'][:400]}")
        print()

    # 3. Edges between UU 11/2020 and UU 40/2007
    r3 = s.run(
        'MATCH (a:Document)-[r]-(b:Document) '
        'WHERE a.doc_id IN ["UU-NASIONAL-11-2020", "UU-NASIONAL-40-2007", "PERPPU-NASIONAL-2-2022"] '
        'AND b.doc_id IN ["UU-NASIONAL-11-2020", "UU-NASIONAL-40-2007", "PERPPU-NASIONAL-2-2022"] '
        'RETURN a.doc_id AS src, type(r) AS rel, b.doc_id AS tgt'
    )
    print("=== Edges between UU 11/2020, UU 40/2007, PERPPU 2/2022 ===")
    for rec in r3:
        print(f"  {rec['src']} --[{rec['rel']}]--> {rec['tgt']}")
    print()

    # 4. Total pasals in UU 11/2020
    r4 = s.run(
        'MATCH (d:Document {doc_id: "UU-NASIONAL-11-2020"})-[:HAS_PASAL]->(p:Pasal) '
        'RETURN count(p) AS cnt'
    )
    print(f"UU-NASIONAL-11-2020 total pasals: {r4.single()['cnt']}")

    # 5. Pasals with Pasal 72 (the dividen interim article)
    r5 = s.run(
        'MATCH (d:Document {doc_id: "UU-NASIONAL-11-2020"})-[:HAS_PASAL]->(p:Pasal) '
        'WHERE p.name CONTAINS "72" '
        'RETURN p.name AS name, p.content AS content'
    )
    print("\n=== UU-NASIONAL-11-2020 Pasal *72* ===")
    for rec in r5:
        print(f"  {rec['name']}: {(rec['content'] or 'NULL')[:500]}")

    # 6. Search Pinecone for dividen in UU 11/2020
    print("\n=== Pinecone: UU-NASIONAL-11-2020 hits for 'dividen interim' ===")
    emb = llm_stance.get_embedding("dividen interim pembagian laba perseroan terbatas pasal 72")
    idx = pinecone_client.get_index()
    r6 = idx.query(vector=emb, top_k=10, include_metadata=True, filter={"doc_id": "UU-NASIONAL-11-2020"})
    for m in r6.get("matches", []):
        meta = m.get("metadata", {})
        print(f"  [{m['score']:.4f}] {meta.get('article_id','?')} / {meta.get('scope','')}:")
        print(f"    {meta.get('content','')[:250]}")
        print()

driver.close()
