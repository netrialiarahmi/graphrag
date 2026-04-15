"""Neo4j graph database connector for legal document graph."""

import os
import functools
import certifi
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Fix SSL on macOS Python 3.10 — set env var BEFORE any SSL connection is made.
# The neo4j driver creates its own ssl.SSLContext which reads SSL_CERT_FILE.
if not os.environ.get("SSL_CERT_FILE"):
    os.environ["SSL_CERT_FILE"] = certifi.where()

# ---------------------------------------------------------------------------
# Streamlit-agnostic caching
# ---------------------------------------------------------------------------
_IN_STREAMLIT = False
if not os.environ.get("GRAPHRAG_STANDALONE"):
    try:
        import streamlit as st
        _IN_STREAMLIT = True
    except ImportError:
        pass


def _cache_resource(func):
    if _IN_STREAMLIT:
        return st.cache_resource(func)
    return functools.lru_cache(maxsize=1)(func)


def _cache_data(**kwargs):
    def decorator(func):
        if _IN_STREAMLIT:
            return st.cache_data(**kwargs)(func)
        return func
    return decorator


load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


@_cache_resource
def get_driver():
    """Create and cache a Neo4j driver instance."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def test_connection() -> bool:
    """Test if Neo4j connection is alive."""
    try:
        driver = get_driver()
        driver.verify_connectivity()
        return True
    except Exception:
        return False


@_cache_data(ttl=3600)
def get_all_documents() -> list[dict]:
    """Fetch all Document nodes with their properties."""
    driver = get_driver()
    query = """
    MATCH (d:Document)
    RETURN d {.doc_id, .judul, .jenis, .tahun, .nomor, .pembentuk, labels: labels(d)} AS doc
    ORDER BY d.doc_id
    """
    with driver.session() as session:
        result = session.run(query)
        return [record["doc"] for record in result]


@_cache_data(ttl=3600)
def get_document_detail(doc_id: str) -> dict:
    """Fetch a single document with all its children (Pasal, Ayat, Diktum)."""
    driver = get_driver()
    # Fetch document
    doc_query = "MATCH (d:Document {doc_id: $doc_id}) RETURN d {.*} AS document"
    # Fetch pasals
    pasal_query = """
    MATCH (d:Document {doc_id: $doc_id})-[:HAS_PASAL]->(p:Pasal)
    RETURN p {.*} AS pasal
    ORDER BY p.name
    """
    # Fetch ayats via pasals
    ayat_query = """
    MATCH (d:Document {doc_id: $doc_id})-[:HAS_PASAL]->(p:Pasal)-[:HAS_AYAT]->(a:Ayat)
    RETURN a {.*, pasal_name: p.name} AS ayat
    ORDER BY p.name, a.name
    """
    # Fetch diktums
    diktum_query = """
    MATCH (d:Document {doc_id: $doc_id})-[:HAS_DIKTUM]->(dk:Diktum)
    RETURN dk {.*} AS diktum
    ORDER BY dk.name
    """
    # Fetch lampiran (attachments)
    lampiran_query = """
    MATCH (d:Document {doc_id: $doc_id})-[:HAS_LAMPIRAN]->(l:Lampiran)
    RETURN l {.*} AS lampiran
    """
    with driver.session() as session:
        doc_result = session.run(doc_query, doc_id=doc_id).single()
        pasals = [r["pasal"] for r in session.run(pasal_query, doc_id=doc_id)]
        ayats = [r["ayat"] for r in session.run(ayat_query, doc_id=doc_id)]
        diktums = [r["diktum"] for r in session.run(diktum_query, doc_id=doc_id)]
        lampirans = [r["lampiran"] for r in session.run(lampiran_query, doc_id=doc_id)]

        if doc_result:
            return {
                "document": doc_result["document"],
                "pasals": pasals,
                "ayats": ayats,
                "diktums": diktums,
                "lampirans": lampirans,
            }
        return {}


@_cache_data(ttl=3600)
def get_document_subgraph(doc_ids: list[str]) -> dict:
    """
    Fetch subgraph for given doc_ids: all nodes (Document, Pasal, Ayat, Diktum)
    and all relationships between them (HAS_PASAL, HAS_AYAT, HAS_DIKTUM, CITES, HIGHER).
    Returns {nodes: [...], edges: [...]}.
    """
    if not doc_ids:
        return {"nodes": [], "edges": []}

    driver = get_driver()
    query = """
    MATCH (d:Document)
    WHERE d.doc_id IN $doc_ids
    OPTIONAL MATCH (d)-[r1:HAS_PASAL]->(p:Pasal)
    OPTIONAL MATCH (p)-[r2:HAS_AYAT]->(a:Ayat)
    OPTIONAL MATCH (d)-[r3:HAS_DIKTUM]->(dk:Diktum)
    WITH collect(DISTINCT d) + collect(DISTINCT p) + collect(DISTINCT a) + collect(DISTINCT dk) AS allNodes
    UNWIND allNodes AS n
    WITH collect(DISTINCT n) AS nodes
    MATCH (d:Document)
    WHERE d.doc_id IN $doc_ids
    OPTIONAL MATCH (d)-[r]-()
    WHERE type(r) IN ['HAS_PASAL', 'HAS_AYAT', 'HAS_DIKTUM', 'CITES', 'HIGHER']
    WITH nodes, collect(DISTINCT {
        source: elementId(startNode(r)),
        target: elementId(endNode(r)),
        type: type(r),
        props: properties(r)
    }) AS rels1
    MATCH (d:Document)
    WHERE d.doc_id IN $doc_ids
    OPTIONAL MATCH (d)-[:HAS_PASAL]->(p:Pasal)-[r]-()
    WHERE type(r) IN ['HAS_AYAT']
    WITH nodes, rels1, collect(DISTINCT {
        source: elementId(startNode(r)),
        target: elementId(endNode(r)),
        type: type(r),
        props: properties(r)
    }) AS rels2
    RETURN [n IN nodes | n {.*, _labels: labels(n), _elementId: elementId(n)}] AS nodes,
           rels1 + rels2 AS edges
    """
    with driver.session() as session:
        result = session.run(query, doc_ids=doc_ids)
        record = result.single()
        if record:
            return {
                "nodes": record["nodes"],
                "edges": [e for e in record["edges"] if e["source"] is not None],
            }
        return {"nodes": [], "edges": []}


@_cache_data(ttl=3600)
def get_citing_documents(doc_id: str, hops: int = 2) -> dict:
    """
    Fetch k-hop subgraph around a document via CITES and HIGHER relationships.
    Returns {nodes: [...], edges: [...]}.
    """
    driver = get_driver()
    query = """
    MATCH (start:Document {doc_id: $doc_id})
    OPTIONAL MATCH path = (start)-[:CITES|HIGHER*1..%d]-(other:Document)
    WITH collect(DISTINCT start {.doc_id, .judul, .jenis, .tahun, .nomor, .pembentuk}) +
         collect(DISTINCT other {.doc_id, .judul, .jenis, .tahun, .nomor, .pembentuk}) AS allNodes,
         collect(path) AS paths
    UNWIND allNodes AS _n
    WITH collect(DISTINCT _n) AS dedupNodes, paths
    WITH [n IN dedupNodes WHERE n IS NOT NULL AND n.doc_id IS NOT NULL] AS nodes, paths
    UNWIND CASE WHEN size(paths) = 0 THEN [null] ELSE paths END AS p
    WITH nodes,
         CASE WHEN p IS NOT NULL
              THEN [r IN relationships(p) | {
                  source_id: startNode(r).doc_id,
                  target_id: endNode(r).doc_id,
                  type: type(r),
                  raw: r.raw
              }]
              ELSE []
         END AS pathEdges
    UNWIND CASE WHEN size(pathEdges) = 0 THEN [null] ELSE pathEdges END AS e
    WITH nodes, collect(DISTINCT e) AS allEdges
    RETURN nodes, [e IN allEdges WHERE e IS NOT NULL] AS edges
    """ % hops

    with driver.session() as session:
        result = session.run(query, doc_id=doc_id)
        record = result.single()
        if record:
            return {
                "nodes": record["nodes"],
                "edges": record["edges"],
            }
        return {"nodes": [], "edges": []}


@_cache_data(ttl=3600)
def get_edges_between(doc_ids: list[str]) -> dict:
    """
    Fetch only the CITES/HIGHER edges that connect documents within the given list.
    Returns {nodes: [...], edges: [...]} with only nodes that are in doc_ids.
    """
    if not doc_ids:
        return {"nodes": [], "edges": []}

    driver = get_driver()
    query = """
    MATCH (d:Document)
    WHERE d.doc_id IN $doc_ids
    WITH collect(d {.doc_id, .judul, .jenis, .tahun, .nomor, .pembentuk}) AS nodes
    OPTIONAL MATCH (a:Document)-[r:CITES|HIGHER]->(b:Document)
    WHERE a.doc_id IN $doc_ids AND b.doc_id IN $doc_ids
    WITH nodes, collect(DISTINCT {
        source_id: a.doc_id,
        target_id: b.doc_id,
        type: type(r),
        raw: r.raw
    }) AS edges
    RETURN nodes, [e IN edges WHERE e.source_id IS NOT NULL] AS edges
    """
    with driver.session() as session:
        result = session.run(query, doc_ids=doc_ids)
        record = result.single()
        if record:
            return {
                "nodes": record["nodes"],
                "edges": record["edges"],
            }
        return {"nodes": [], "edges": []}


@_cache_data(ttl=3600)
def get_graph_overview() -> dict:
    """
    Fetch the full document-level graph: all Document nodes
    and CITES / HIGHER edges between them.
    """
    driver = get_driver()
    query = """
    MATCH (d:Document)
    WITH collect(d {.doc_id, .judul, .jenis, .tahun, .nomor, .pembentuk}) AS nodes
    OPTIONAL MATCH (a:Document)-[r:CITES|HIGHER]->(b:Document)
    WITH nodes, collect(DISTINCT {
        source_id: a.doc_id,
        target_id: b.doc_id,
        type: type(r),
        raw: r.raw
    }) AS edges
    RETURN nodes, [e IN edges WHERE e.source_id IS NOT NULL] AS edges
    """
    with driver.session() as session:
        result = session.run(query)
        record = result.single()
        if record:
            return {
                "nodes": record["nodes"],
                "edges": record["edges"],
            }
        return {"nodes": [], "edges": []}


@_cache_data(ttl=3600)
def get_schema_info() -> dict:
    """Get counts of each node label and relationship type."""
    driver = get_driver()
    node_query = """
    CALL db.labels() YIELD label
    CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) AS cnt', {}) YIELD value
    RETURN label, value.cnt AS count
    """
    # Fallback simpler query
    simple_query = """
    MATCH (n)
    WITH labels(n) AS lbls
    UNWIND lbls AS label
    RETURN label, count(*) AS count
    ORDER BY count DESC
    """
    rel_query = """
    MATCH ()-[r]->()
    RETURN type(r) AS type, count(*) AS count
    ORDER BY count DESC
    """
    with driver.session() as session:
        try:
            node_result = session.run(simple_query)
            node_counts = {r["label"]: r["count"] for r in node_result}
        except Exception:
            node_counts = {}

        try:
            rel_result = session.run(rel_query)
            rel_counts = {r["type"]: r["count"] for r in rel_result}
        except Exception:
            rel_counts = {}

    return {"node_counts": node_counts, "rel_counts": rel_counts}


@_cache_data(ttl=3600)
def get_related_documents(doc_id: str, limit: int = 3) -> list[dict]:
    """
    Fetch documents directly related to the given doc_id via CITES or HIGHER.
    Returns up to `limit` related document nodes.
    """
    driver = get_driver()
    query = """
    MATCH (d:Document {doc_id: $doc_id})-[:CITES|HIGHER]-(other:Document)
    WHERE other.doc_id <> $doc_id
    RETURN DISTINCT other {.doc_id, .judul, .jenis, .tahun, .nomor, .pembentuk} AS doc
    LIMIT $limit
    """
    with driver.session() as session:
        result = session.run(query, doc_id=doc_id, limit=limit)
        return [record["doc"] for record in result]


def search_lampiran_content(keywords: list[str], max_docs: int = 5) -> list[str]:
    """Search Lampiran nodes by keyword matching and return doc_ids ranked by relevance.

    This enables discovery of documents whose primary content lives in
    Lampiran (attachments) rather than in Pasal/Ayat text.
    """
    if not keywords:
        return []
    driver = get_driver()
    # Build a WHERE clause that counts keyword hits
    conditions = " + ".join(
        [f"CASE WHEN toLower(l.content) CONTAINS toLower($kw{i}) THEN 1 ELSE 0 END"
         for i in range(len(keywords))]
    )
    query = f"""
    MATCH (d:Document)-[:HAS_LAMPIRAN]->(l:Lampiran)
    WHERE l.content IS NOT NULL AND size(l.content) > 30
    WITH d.doc_id AS doc_id, l, ({conditions}) AS hits
    WHERE hits > 0
    RETURN doc_id, sum(hits) AS total_hits, count(l) AS matching_lampiran
    ORDER BY total_hits DESC, matching_lampiran DESC
    LIMIT $max_docs
    """
    params = {f"kw{i}": kw for i, kw in enumerate(keywords)}
    params["max_docs"] = max_docs
    with driver.session() as session:
        result = session.run(query, **params)
        return [r["doc_id"] for r in result]


@_cache_data(ttl=3600)
def get_all_document_pairs() -> list[dict]:
    """
    Fetch all CITES/HIGHER relationship pairs between documents.
    Returns list of {source_id, target_id, type, raw}.
    """
    driver = get_driver()
    query = """
    MATCH (a:Document)-[r:CITES|HIGHER]->(b:Document)
    RETURN DISTINCT a.doc_id AS source_id, b.doc_id AS target_id, type(r) AS type, r.raw AS raw
    ORDER BY a.doc_id, b.doc_id
    """
    with driver.session() as session:
        result = session.run(query)
        return [dict(record) for record in result]
