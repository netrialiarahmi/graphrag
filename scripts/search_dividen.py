"""Search Neo4j for dividen-related content in UU 40/2007 and Perppu 2/2022."""
import os, sys
os.environ['GRAPHRAG_STANDALONE'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()
from utils import neo4j_client

driver = neo4j_client.get_driver()
with driver.session() as s:
    # 1. UU 40/2007 pasals with dividen
    r = s.run(
        'MATCH (d:Document {doc_id: "UU-NASIONAL-40-2007"})-[:HAS_PASAL]->(p:Pasal) '
        'WHERE toLower(p.content) CONTAINS "dividen" '
        'RETURN p.name AS name, substring(p.content, 0, 400) AS content ORDER BY p.name'
    )
    print("=== UU-NASIONAL-40-2007 pasals with 'dividen' ===")
    for rec in r:
        print(f"  {rec['name']}: {rec['content'][:200]}...")
    print()

    # 2. UU 40/2007 ayats with dividen
    r2 = s.run(
        'MATCH (d:Document {doc_id: "UU-NASIONAL-40-2007"})-[:HAS_PASAL]->(p:Pasal)-[:HAS_AYAT]->(a:Ayat) '
        'WHERE toLower(a.content) CONTAINS "dividen" '
        'RETURN p.name AS pasal, a.name AS ayat, substring(a.content, 0, 400) AS content ORDER BY p.name, a.name'
    )
    print("=== UU-NASIONAL-40-2007 ayats with 'dividen' ===")
    for rec in r2:
        print(f"  {rec['pasal']}/{rec['ayat']}: {rec['content'][:200]}...")
    print()

    # 3. PERPPU 2/2022 pasals with dividen
    r3 = s.run(
        'MATCH (d:Document {doc_id: "PERPPU-NASIONAL-2-2022"})-[:HAS_PASAL]->(p:Pasal) '
        'WHERE toLower(p.content) CONTAINS "dividen" '
        'RETURN p.name AS name, substring(p.content, 0, 400) AS content ORDER BY p.name'
    )
    print("=== PERPPU-NASIONAL-2-2022 pasals with 'dividen' ===")
    for rec in r3:
        print(f"  {rec['name']}: {rec['content'][:200]}...")
    print()

    # 4. PERPPU 2/2022 ayats with dividen
    r4 = s.run(
        'MATCH (d:Document {doc_id: "PERPPU-NASIONAL-2-2022"})-[:HAS_PASAL]->(p:Pasal)-[:HAS_AYAT]->(a:Ayat) '
        'WHERE toLower(a.content) CONTAINS "dividen" '
        'RETURN p.name AS pasal, a.name AS ayat, substring(a.content, 0, 400) AS content ORDER BY p.name, a.name'
    )
    print("=== PERPPU-NASIONAL-2-2022 ayats with 'dividen' ===")
    for rec in r4:
        print(f"  {rec['pasal']}/{rec['ayat']}: {rec['content'][:200]}...")
    print()

    # 5. Check what Pasal 72 contains
    r5 = s.run(
        'MATCH (d:Document {doc_id: "UU-NASIONAL-40-2007"})-[:HAS_PASAL]->(p:Pasal) '
        'WHERE p.name CONTAINS "72" '
        'RETURN p.name AS name, p.content AS content'
    )
    print("=== UU-NASIONAL-40-2007 Pasal 72 ===")
    for rec in r5:
        print(f"  {rec['name']}: {rec['content']}")
    print()

    # 6. Check edges between the two docs
    r6 = s.run(
        'MATCH (a:Document)-[r]-(b:Document) '
        'WHERE a.doc_id IN ["UU-NASIONAL-40-2007", "PERPPU-NASIONAL-2-2022"] '
        'AND b.doc_id IN ["UU-NASIONAL-40-2007", "PERPPU-NASIONAL-2-2022"] '
        'RETURN a.doc_id AS src, type(r) AS rel, b.doc_id AS tgt'
    )
    print("=== Edges between UU 40/2007 and PERPPU 2/2022 ===")
    for rec in r6:
        print(f"  {rec['src']} --[{rec['rel']}]--> {rec['tgt']}")
    print()

    # 7. Also search Pinecone for dividen interim in these docs
    print("=== Pinecone search for 'dividen interim' ===")
    # 8. List all pasals in UU 40/2007
    r8 = s.run(
        'MATCH (d:Document {doc_id: "UU-NASIONAL-40-2007"})-[:HAS_PASAL]->(p:Pasal) '
        'RETURN p.name AS name ORDER BY p.name'
    )
    pasals = [rec['name'] for rec in r8]
    print(f"=== UU-NASIONAL-40-2007 total pasals: {len(pasals)} ===")
    print(f"  Names: {pasals[:20]}...")
    print()

    # 9. List all pasals in PERPPU 2/2022
    r9 = s.run(
        'MATCH (d:Document {doc_id: "PERPPU-NASIONAL-2-2022"})-[:HAS_PASAL]->(p:Pasal) '
        'RETURN p.name AS name ORDER BY p.name'
    )
    pasals2 = [rec['name'] for rec in r9]
    print(f"=== PERPPU-NASIONAL-2-2022 total pasals: {len(pasals2)} ===")
    print(f"  Names: {pasals2[:30]}")
    print()

    # 10. Check Pasal 14 of PERPPU 2/2022 (the one that changes UU 40/2007)
    r10 = s.run(
        'MATCH (d:Document {doc_id: "PERPPU-NASIONAL-2-2022"})-[:HAS_PASAL]->(p:Pasal) '
        'WHERE p.name CONTAINS "14" '
        'RETURN p.name AS name, p.content AS content'
    )
    print("=== PERPPU-NASIONAL-2-2022 Pasal 14 ===")
    for rec in r10:
        print(f"  {rec['name']}: {rec['content'][:500] if rec['content'] else 'NULL'}")
    print()

    # 11. Search ANY doc for dividen interim
    r11 = s.run(
        'MATCH (d:Document)-[:HAS_PASAL]->(p:Pasal) '
        'WHERE toLower(p.content) CONTAINS "dividen interim" '
        'RETURN d.doc_id AS doc, p.name AS name, substring(p.content, 0, 300) AS content '
        'LIMIT 10'
    )
    print("=== ANY doc with 'dividen interim' in pasals ===")
    for rec in r11:
        print(f"  {rec['doc']}/{rec['name']}: {rec['content'][:200]}...")
    print()

    r12 = s.run(
        'MATCH (d:Document)-[:HAS_PASAL]->(p:Pasal)-[:HAS_AYAT]->(a:Ayat) '
        'WHERE toLower(a.content) CONTAINS "dividen interim" '
        'RETURN d.doc_id AS doc, p.name AS pasal, a.name AS ayat, substring(a.content, 0, 300) AS content '
        'LIMIT 10'
    )
    print("=== ANY doc with 'dividen interim' in ayats ===")
    for rec in r12:
        print(f"  {rec['doc']}/{rec['pasal']}/{rec['ayat']}: {rec['content'][:200]}...")
driver.close()
