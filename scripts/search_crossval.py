"""Deep check: what dividen ayats exist and which Pasal 4 are relevant."""
import os, sys
os.environ['GRAPHRAG_STANDALONE'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv; load_dotenv()
from utils import neo4j_client

driver = neo4j_client.get_driver()
with driver.session() as s:
    # 1. Full content of UU 11/2020 Pasal 4 ayats with dividen
    r = s.run(
        'MATCH (d:Document {doc_id: "UU-NASIONAL-11-2020"})-[:HAS_PASAL]->(p:Pasal)-[:HAS_AYAT]->(a:Ayat) '
        'WHERE toLower(a.content) CONTAINS "dividen" '
        'RETURN p.name AS pasal, a.name AS ayat, a.content AS content ORDER BY p.name, a.name'
    )
    print("=== UU 11/2020 all ayats mentioning 'dividen' (full) ===")
    count = 0
    for rec in r:
        count += 1
        print(f"\n--- {rec['pasal']}/{rec['ayat']} ---")
        print(rec['content'][:600])
    print(f"\nTotal: {count}")

    # 2. Same for PERPPU 2/2022
    r2 = s.run(
        'MATCH (d:Document {doc_id: "PERPPU-NASIONAL-2-2022"})-[:HAS_PASAL]->(p:Pasal)-[:HAS_AYAT]->(a:Ayat) '
        'WHERE toLower(a.content) CONTAINS "dividen" '
        'RETURN p.name AS pasal, a.name AS ayat, a.content AS content ORDER BY p.name, a.name'
    )
    print("\n=== PERPPU 2/2022 all ayats mentioning 'dividen' (full) ===")
    count2 = 0
    for rec in r2:
        count2 += 1
        print(f"\n--- {rec['pasal']}/{rec['ayat']} ---")
        print(rec['content'][:600])
    print(f"\nTotal: {count2}")

    # 3. Check how many ayats have NULL content in each doc
    for did in ["UU-NASIONAL-11-2020", "PERPPU-NASIONAL-2-2022", "UU-NASIONAL-40-2007"]:
        r3 = s.run(
            'MATCH (d:Document {doc_id: $did})-[:HAS_PASAL]->(p:Pasal) '
            'WHERE p.content IS NULL OR p.content = "" '
            'RETURN count(p) AS null_cnt',
            did=did
        )
        null_cnt = r3.single()['null_cnt']
        r4 = s.run(
            'MATCH (d:Document {doc_id: $did})-[:HAS_PASAL]->(p:Pasal) '
            'WHERE p.content IS NOT NULL AND p.content <> "" '
            'RETURN count(p) AS has_cnt',
            did=did
        )
        has_cnt = r4.single()['has_cnt']
        print(f"\n{did}: {has_cnt} pasals WITH content, {null_cnt} pasals with NULL content")

    # 4. Check if there's a "dividen interim" specifically anywhere
    r5 = s.run(
        'MATCH (d:Document)-[:HAS_PASAL]->(p:Pasal)-[:HAS_AYAT]->(a:Ayat) '
        'WHERE toLower(a.content) CONTAINS "dividen interim" '
        'RETURN d.doc_id AS doc, p.name AS pasal, a.name AS ayat, substring(a.content, 0, 300) AS content '
        'LIMIT 20'
    )
    print("\n=== ANY doc with exact 'dividen interim' in ayats ===")
    for rec in r5:
        print(f"  {rec['doc']}/{rec['pasal']}/{rec['ayat']}: {rec['content'][:200]}")

    # 5. Cross-check: documents that CITE UU 40/2007 or are HIGHER
    r6 = s.run(
        'MATCH (a:Document)-[r]->(b:Document {doc_id: "UU-NASIONAL-40-2007"}) '
        'RETURN a.doc_id AS src, type(r) AS rel LIMIT 20'
    )
    print("\n=== Docs citing/linked to UU 40/2007 ===")
    for rec in r6:
        print(f"  {rec['src']} --[{rec['rel']}]--> UU-NASIONAL-40-2007")

    r7 = s.run(
        'MATCH (b:Document {doc_id: "UU-NASIONAL-40-2007"})-[r]->(a:Document) '
        'RETURN a.doc_id AS tgt, type(r) AS rel LIMIT 20'
    )
    for rec in r7:
        print(f"  UU-NASIONAL-40-2007 --[{rec['rel']}]--> {rec['tgt']}")

driver.close()
