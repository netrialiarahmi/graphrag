"""
Knowledge Map – Interactive Legal Document Explorer
Open Knowledge Maps-style visualization for Indonesian regulations.

Query → Semantic Search → Topic Clustering → Interactive Graph → PDF Access
"""

import streamlit as st
import os, sys, json, re, time
from datetime import datetime

# -- Resolve project root so shared/ is importable
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# -- Streamlit Cloud: inject secrets into env vars
try:
    if hasattr(st, "secrets") and len(st.secrets):
        for key, value in st.secrets.items():
            if isinstance(value, str):
                os.environ.setdefault(key, value)
except Exception:
    pass

from dotenv import load_dotenv
load_dotenv()

import boto3
from streamlit_agraph import agraph, Node, Edge, Config
from shared import neo4j_client, pinecone_client, llm_stance

# -- Page Config
st.set_page_config(
    page_title="Knowledge Map – GraphRAG",
    page_icon="\U0001f5fa\ufe0f",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==============================================================================
# CSS
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }

.block-container {
    max-width: 1400px !important;
    padding: 1rem 2rem 4rem !important;
}

/* Hero search */
.km-hero {
    text-align: center; padding: 3rem 1rem 1.5rem;
}
.km-hero h1 {
    font-size: 2.2rem; font-weight: 800; color: #1a1a2e;
    letter-spacing: -0.5px; margin-bottom: 0.3rem;
}
.km-hero h1 span {
    background: linear-gradient(135deg, #2563eb, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.km-hero p {
    font-size: 0.95rem; color: #64748b; max-width: 560px;
    margin: 0 auto; line-height: 1.6;
}

/* Cluster cards */
.cluster-card {
    border: 2px solid #e2e8f0; border-radius: 16px;
    padding: 18px 20px; margin-bottom: 12px;
    background: #ffffff; transition: all 0.25s ease;
}
.cluster-card:hover {
    border-color: #2563eb; box-shadow: 0 4px 20px rgba(37,99,235,0.12);
    transform: translateY(-2px);
}
.cluster-title {
    font-size: 1rem; font-weight: 700; color: #1e3a5f;
    margin-bottom: 6px;
}
.cluster-count {
    font-size: 0.75rem; font-weight: 600; color: #2563eb;
    background: #eff6ff; padding: 2px 8px; border-radius: 20px;
    display: inline-block; margin-bottom: 8px;
}
.cluster-docs {
    font-size: 0.8rem; color: #64748b; line-height: 1.5;
}

/* Stats bar */
.stats-bar {
    display: flex; gap: 24px; justify-content: center;
    padding: 12px 0; margin-bottom: 8px;
}
.stat-item { text-align: center; }
.stat-num { font-size: 1.5rem; font-weight: 800; color: #2563eb; }
.stat-label {
    font-size: 0.72rem; color: #94a3b8; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.5px;
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
    .stApp, [data-testid="stAppViewContainer"],
    section[data-testid="stMain"] { background-color: #0c1322 !important; }
    .km-hero h1 { color: #f1f5f9; }
    .km-hero p { color: #94a3b8; }
    .cluster-card { background: #1e293b; border-color: #334155; }
    .cluster-card:hover { border-color: #2563eb; }
    .cluster-title { color: #f1f5f9; }
    .cluster-docs { color: #94a3b8; }
}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SESSION STATE
# ==============================================================================
for _k, _v in {
    "km_results": None,
    "km_query": "",
    "km_selected_node": None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ==============================================================================
# HELPERS
# ==============================================================================

CLUSTER_COLORS = [
    "#2563eb", "#059669", "#d97706", "#7c3aed", "#dc2626",
    "#0891b2", "#c026d3", "#65a30d", "#ea580c", "#6366f1",
]


def _get_doc_short_label(doc_id: str) -> str:
    parts = doc_id.split("-")
    if len(parts) >= 4:
        return f"{parts[0]} {parts[2]}/{parts[3]}"
    return doc_id[:30]


def _get_doc_type_from_id(doc_id: str) -> str:
    prefix = doc_id.split("-")[0].upper() if "-" in doc_id else ""
    if prefix == "UU":
        return "Undang-Undang"
    elif prefix == "PP":
        return "Peraturan Pemerintah"
    elif prefix.startswith("PERMEN"):
        return "Peraturan Menteri"
    elif prefix in ("PERPRES", "KEPPRES"):
        return "Peraturan Presiden"
    elif prefix == "PERPPU":
        return "Perppu"
    return "Regulasi"


@st.cache_data(ttl=300, show_spinner=False)
def _s3_presigned_url(doc_id: str) -> str | None:
    try:
        region = os.getenv("AWS_REGION", "ap-southeast-3")
        bucket = os.getenv("S3_BUCKET", "s3-lexport-dev-v1")
        directory = os.getenv("S3_DIRECTORY", "neo4j-dev")
        key = f"{directory}/{doc_id}.pdf"
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region,
            endpoint_url=f"https://s3.{region}.amazonaws.com",
        )
        s3.head_object(Bucket=bucket, Key=key)
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=3600,
        )
    except Exception:
        return None


def _cluster_documents(query: str, doc_metas: list[dict]) -> list[dict]:
    """Use LLM to cluster documents into topic groups."""
    if not doc_metas:
        return []

    doc_list = "\n".join(
        f"- {d['doc_id']}: {d.get('judul', d['doc_id'])} ({_get_doc_type_from_id(d['doc_id'])})"
        for d in doc_metas
    )

    client = llm_stance.get_llm_client()
    model = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4")

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=1000,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Kamu adalah ahli hukum Indonesia. Kelompokkan dokumen regulasi berikut "
                        "ke dalam 3-7 cluster topik berdasarkan kesamaan tema/bidang hukumnya.\n\n"
                        "WAJIB kembalikan JSON array, TANPA markdown fence:\n"
                        '[{"topic": "Nama Topik Singkat", "description": "1 kalimat deskripsi", "doc_ids": ["DOC-ID-1", "DOC-ID-2"]}]\n\n'
                        "Aturan:\n"
                        "- Setiap dokumen HARUS masuk tepat 1 cluster\n"
                        "- Nama topik: 2-5 kata, bahasa Indonesia\n"
                        "- Jika dokumen < 4, buat 1-2 cluster saja\n"
                        "- doc_ids harus persis sama dengan yang diberikan"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Query pengguna: {query}\n\nDokumen:\n{doc_list}",
                },
            ],
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        clusters = json.loads(raw)
        if isinstance(clusters, list) and all("topic" in c and "doc_ids" in c for c in clusters):
            return clusters
    except Exception:
        pass

    # Fallback: single cluster
    return [{
        "topic": "Hasil Pencarian",
        "description": f"Dokumen terkait: {query}",
        "doc_ids": [d["doc_id"] for d in doc_metas],
    }]


def _search_and_build(query: str) -> dict:
    """Execute search pipeline and return structured results."""
    t0 = time.time()

    # 1. Embed query & semantic search
    embedding = llm_stance.get_embedding(query)
    vdb_hits = pinecone_client.semantic_search(embedding, top_k=30)

    # 2. Extract unique doc_ids (ordered by best score)
    seen = {}
    for hit in vdb_hits:
        did = hit.get("doc_id", "")
        if did and did not in seen:
            seen[did] = hit.get("score", 0)
    doc_ids_sorted = sorted(seen.keys(), key=lambda d: seen[d], reverse=True)[:15]

    # 3. Fetch Neo4j metadata
    all_docs = neo4j_client.get_all_documents()
    doc_map = {d["doc_id"]: d for d in all_docs if d.get("doc_id")}
    doc_metas = []
    for did in doc_ids_sorted:
        meta = doc_map.get(did, {"doc_id": did})
        meta["doc_id"] = did
        meta["vdb_score"] = seen.get(did, 0)
        doc_metas.append(meta)

    # 4. Fetch relationships
    edges_data = {"nodes": [], "edges": []}
    if len(doc_ids_sorted) >= 2 and neo4j_client.test_connection():
        try:
            edges_data = neo4j_client.get_edges_between(doc_ids_sorted)
        except Exception:
            pass

    # 5. LLM topic clustering
    clusters = _cluster_documents(query, doc_metas)

    # 6. Collect chunks per doc
    chunks_by_doc = {}
    for hit in vdb_hits:
        did = hit.get("doc_id", "")
        if did in seen:
            chunks_by_doc.setdefault(did, []).append(hit)

    return {
        "query": query,
        "doc_metas": doc_metas,
        "doc_map": {d["doc_id"]: d for d in doc_metas},
        "clusters": clusters,
        "edges": edges_data.get("edges", []),
        "chunks_by_doc": chunks_by_doc,
        "latency": round(time.time() - t0, 1),
        "total_docs": len(doc_ids_sorted),
        "total_chunks": len(vdb_hits),
    }


def _build_graph(results: dict) -> tuple[list, list]:
    """Build agraph Nodes and Edges from search results."""
    nodes = []
    edges = []
    clusters = results["clusters"]
    doc_map = results["doc_map"]

    doc_to_cluster = {}
    for ci, cluster in enumerate(clusters):
        for did in cluster.get("doc_ids", []):
            doc_to_cluster[did] = ci

    # Topic cluster nodes (large)
    for ci, cluster in enumerate(clusters):
        color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
        n_docs = len(cluster.get("doc_ids", []))
        nodes.append(Node(
            id=f"cluster_{ci}",
            label=cluster["topic"],
            size=35 + n_docs * 5,
            color=color,
            font={"size": 16, "color": "#ffffff", "bold": True},
            shape="dot",
            opacity=0.85,
            borderWidth=3,
            borderWidthSelected=5,
            title=f"{cluster['topic']}\n{cluster.get('description', '')}\n{n_docs} dokumen",
        ))

    # Document nodes (small, bordered)
    for did, meta in doc_map.items():
        ci = doc_to_cluster.get(did, 0)
        color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
        label = _get_doc_short_label(did)
        judul = meta.get("judul", did)
        score = meta.get("vdb_score", 0)
        doc_type = _get_doc_type_from_id(did)

        nodes.append(Node(
            id=did,
            label=label,
            size=18 + score * 12,
            color={"background": "#ffffff", "border": color,
                   "highlight": {"background": color, "border": color}},
            font={"size": 11, "color": "#334155"},
            shape="dot",
            borderWidth=2,
            borderWidthSelected=4,
            title=f"{did}\n{judul}\n{doc_type}\nRelevance: {score:.2f}",
        ))

        # Edge: doc -> cluster
        edges.append(Edge(
            source=did,
            target=f"cluster_{ci}",
            color=color,
            width=1.5,
            smooth={"type": "continuous"},
        ))

    # Cross-document relationship edges (CITES, HIGHER)
    existing_docs = set(doc_map.keys())
    for edge in results.get("edges", []):
        src = edge.get("source_id", "")
        tgt = edge.get("target_id", "")
        rel = edge.get("type", "CITES")
        if src in existing_docs and tgt in existing_docs:
            edge_color = "#2563eb" if rel == "CITES" else "#94a3b8"
            edges.append(Edge(
                source=src,
                target=tgt,
                label=rel,
                color=edge_color,
                width=1,
                dashes=rel == "HIGHER",
                font={"size": 8, "color": "#94a3b8"},
                smooth={"type": "curvedCW", "roundness": 0.2},
            ))

    return nodes, edges


# ==============================================================================
# UI – HERO / SEARCH
# ==============================================================================

st.markdown(
    '<div class="km-hero">'
    '<h1>\U0001f5fa\ufe0f Knowledge <span>Map</span></h1>'
    "<p>Eksplorasi visual regulasi Indonesia. Masukkan topik hukum dan lihat "
    "peta pengetahuan interaktif \u2014 klik dokumen untuk membaca PDF.</p>"
    "</div>",
    unsafe_allow_html=True,
)

col_pad1, col_search, col_btn, col_pad2 = st.columns([1, 5, 1, 1])
with col_search:
    query_input = st.text_input(
        "Cari topik hukum",
        placeholder="Contoh: perizinan berusaha, dividen perseroan, ketenagakerjaan...",
        label_visibility="collapsed",
    )
with col_btn:
    search_clicked = st.button("\U0001f50d Cari", type="primary", use_container_width=True)

if search_clicked and query_input.strip():
    st.session_state.km_query = query_input.strip()
    st.session_state.km_selected_node = None
    with st.spinner("Membangun peta pengetahuan..."):
        st.session_state.km_results = _search_and_build(query_input.strip())


# ==============================================================================
# UI – RESULTS
# ==============================================================================

results = st.session_state.km_results

if results:
    # Stats bar
    st.markdown(
        f'<div class="stats-bar">'
        f'<div class="stat-item"><div class="stat-num">{results["total_docs"]}</div>'
        f'<div class="stat-label">Dokumen</div></div>'
        f'<div class="stat-item"><div class="stat-num">{len(results["clusters"])}</div>'
        f'<div class="stat-label">Cluster Topik</div></div>'
        f'<div class="stat-item"><div class="stat-num">{results["total_chunks"]}</div>'
        f'<div class="stat-label">Fragmen</div></div>'
        f'<div class="stat-item"><div class="stat-num">{results["latency"]}s</div>'
        f'<div class="stat-label">Latensi</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Two-column: graph + detail panel
    col_graph, col_detail = st.columns([3, 2])

    with col_graph:
        st.markdown("##### \U0001f5fa\ufe0f Peta Pengetahuan")
        st.caption("Klik node untuk melihat detail. Lingkaran besar = cluster topik, kecil = dokumen.")

        graph_nodes, graph_edges = _build_graph(results)

        config = Config(
            width="100%",
            height=620,
            directed=True,
            physics=True,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#2563eb",
            collapsible=False,
            node={"labelProperty": "label"},
            link={"labelProperty": "label", "renderLabel": True},
        )

        selected_node = agraph(
            nodes=graph_nodes,
            edges=graph_edges,
            config=config,
        )

        if selected_node:
            st.session_state.km_selected_node = selected_node

    with col_detail:
        selected = st.session_state.km_selected_node

        if selected and selected.startswith("cluster_"):
            # Cluster detail
            ci = int(selected.replace("cluster_", ""))
            if ci < len(results["clusters"]):
                cluster = results["clusters"][ci]
                color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]

                st.markdown(f"##### \U0001f4c1 {cluster['topic']}")
                st.caption(cluster.get("description", ""))

                for did in cluster.get("doc_ids", []):
                    meta = results["doc_map"].get(did, {})
                    label = _get_doc_short_label(did)
                    judul = meta.get("judul", did)

                    with st.container(border=True):
                        st.markdown(f"**{label}**")
                        st.caption(judul)

                        url = _s3_presigned_url(did)
                        if url:
                            st.link_button("\U0001f4c4 Buka PDF", url, use_container_width=True)
                        else:
                            st.warning("PDF tidak tersedia di S3", icon="\u26a0\ufe0f")

                        if st.button("Lihat Detail", key=f"detail_{did}"):
                            st.session_state.km_selected_node = did
                            st.rerun()

        elif selected and not selected.startswith("cluster_"):
            # Document detail
            did = selected
            meta = results["doc_map"].get(did, {"doc_id": did})
            label = _get_doc_short_label(did)
            judul = meta.get("judul", did)
            doc_type = _get_doc_type_from_id(did)
            score = meta.get("vdb_score", 0)

            st.markdown(f"##### \U0001f4dc {label}")
            st.caption(f"{doc_type} \u2014 Relevansi: {score:.2f}")
            st.markdown(f"**{judul}**")

            cols = st.columns(3)
            with cols[0]:
                st.metric("Jenis", meta.get("jenis", "-"))
            with cols[1]:
                st.metric("Tahun", meta.get("tahun", "-"))
            with cols[2]:
                st.metric("Nomor", meta.get("nomor", "-"))

            # PDF
            url = _s3_presigned_url(did)
            if url:
                st.link_button("\U0001f4c4 Buka PDF", url, use_container_width=True, type="primary")
            else:
                st.warning("PDF tidak tersedia di S3", icon="\u26a0\ufe0f")

            # Relevant chunks
            chunks = results.get("chunks_by_doc", {}).get(did, [])
            if chunks:
                st.markdown("---")
                st.markdown(f"**Kutipan Relevan** ({len(chunks)} fragmen)")
                for i, chunk in enumerate(chunks[:5]):
                    content = chunk.get("content", "")[:400]
                    article = chunk.get("article_id", "")
                    with st.expander(
                        f"\U0001f4cb {article}" if article else f"\U0001f4cb Fragmen {i+1}",
                        expanded=i == 0,
                    ):
                        st.markdown(
                            f"<div style='font-size:0.82rem;line-height:1.6;color:#334155;'>"
                            f"{content}</div>",
                            unsafe_allow_html=True,
                        )
                        st.caption(f"Skor: {chunk.get('score', 0):.3f}")

            # Related docs (edges)
            related = []
            for edge in results.get("edges", []):
                src = edge.get("source_id", "")
                tgt = edge.get("target_id", "")
                rel = edge.get("type", "")
                if src == did:
                    related.append((tgt, rel, "\u2192"))
                elif tgt == did:
                    related.append((src, rel, "\u2190"))

            if related:
                st.markdown("---")
                st.markdown(f"**Relasi Hukum** ({len(related)})")
                for rdoc, rel, direction in related:
                    rlabel = _get_doc_short_label(rdoc)
                    rel_icon = "\U0001f517" if rel == "CITES" else "\u2b06\ufe0f"
                    st.markdown(f"{rel_icon} {direction} **{rlabel}** ({rel})")
                    if st.button(f"Lihat {rlabel}", key=f"nav_{rdoc}"):
                        st.session_state.km_selected_node = rdoc
                        st.rerun()

        else:
            # No selection – cluster overview
            st.markdown("##### \U0001f4ca Cluster Topik")
            st.caption("Klik node di peta atau pilih cluster di bawah.")

            for ci, cluster in enumerate(results["clusters"]):
                color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
                n_docs = len(cluster.get("doc_ids", []))
                doc_labels = ", ".join(
                    _get_doc_short_label(d) for d in cluster.get("doc_ids", [])[:5]
                )
                if n_docs > 5:
                    doc_labels += f", +{n_docs - 5} lainnya"

                st.markdown(
                    f'<div class="cluster-card">'
                    f'<div class="cluster-title" style="color:{color};">\u25cf {cluster["topic"]}</div>'
                    f'<div class="cluster-count">{n_docs} dokumen</div>'
                    f'<div class="cluster-docs">{doc_labels}</div>'
                    f'<div class="cluster-docs" style="margin-top:4px;font-style:italic;">'
                    f'{cluster.get("description", "")}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

                if st.button("Lihat Cluster", key=f"cluster_btn_{ci}", use_container_width=True):
                    st.session_state.km_selected_node = f"cluster_{ci}"
                    st.rerun()

elif not results and not query_input:
    # Example queries
    st.markdown("---")
    st.markdown("##### Contoh Pencarian")
    examples = [
        "Perizinan berusaha dan OSS",
        "Ketenagakerjaan dan PHK",
        "Perseroan terbatas dan dividen",
        "Pertambangan mineral dan batubara",
        "Cipta kerja dan kemudahan investasi",
    ]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        with cols[i]:
            if st.button(f"\U0001f50d {ex}", key=f"ex_{i}", use_container_width=True):
                st.session_state.km_query = ex
                st.session_state.km_selected_node = None
                with st.spinner("Membangun peta pengetahuan..."):
                    st.session_state.km_results = _search_and_build(ex)
                st.rerun()

# Footer
st.markdown("---")
st.caption("Knowledge Map \u2014 GraphRAG Legal AI \u00b7 Neo4j + Pinecone + LLM Topic Clustering")
