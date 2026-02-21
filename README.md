# GraphRAG -- Legal Document Retrieval Benchmark

A benchmark framework for evaluating information retrieval performance on Indonesian legal documents, comparing Vector Database (VDB) retrieval against Graph-augmented Retrieval (GraphRAG).

---

## Project Structure

```
graphrag/
  app.py                     # Streamlit web application (read-only result viewer)
  run_benchmark_v3.py        # Retrieval benchmark CLI script
  run_kausalitas.py          # Kausalitas (legal consistency) analysis CLI script
  requirements.txt           # Python dependencies
  benchmark/                 # Ground-truth XLSX datasets
    QA 100 (test-all-sector).xlsx
    govnetic_qa_complete_50 (business).xlsx
  output/
    retrieval/
      detailed retrieval/    # Per-question retrieval results (CSV)
      metrics/               # Aggregate metric summaries (CSV)
    kausalitas/              # Kausalitas analysis results
  utils/
    benchmark_helpers.py     # Document parsing, alias matching, scoring helpers
    neo4j_client.py          # Neo4j Aura graph database client
    pinecone_client.py       # Pinecone vector database client
    llm_stance.py            # Embedding and LLM inference utilities
    graph_viz.py             # Graph visualization helpers
```

---

## Benchmark v3 -- Methodology

### Overview

The retrieval benchmark evaluates how effectively the system retrieves relevant legal documents given a natural language question. Each question in the ground-truth dataset is annotated with evidence text referencing specific Indonesian regulations (e.g., "PP 34/2021", "Permen PUPR 8/2022"). The benchmark parses these references into structured document identifiers, queries the retrieval systems, and measures recall and precision.

Two retrieval strategies are compared:

1. **VDB-only**: Semantic search against Pinecone using dense embeddings from Indo-LegalBERT-V3.
2. **GraphRAG**: VDB retrieval augmented with graph-based expansion via Neo4j, traversing citation (CITES) and hierarchical (HIGHER) relationships between legal documents.

### Pipeline

The benchmark executes the following steps for each question:

**Step 1 -- Ground-Truth Extraction**

The system parses the evidence column from the XLSX dataset to identify all referenced legal documents. A regex-based parser matches patterns such as "UU 2/2017", "Permen PUPR 8/2022", or "SK Dirjen BK 2022" and maps them to canonical document identifiers using a normalised type map.

Additionally, the question text itself is parsed for explicit regulation references (e.g., "menurut PP 34/2021"). These are injected into the ground-truth set, as questions that explicitly name a regulation should expect that regulation to appear in the retrieval results.

**Step 2 -- Embedding and VDB Retrieval**

The question text is embedded using the Govnetic/Indo-LegalBERT-V3 model (1024-dimensional vectors). The embedding is used to query Pinecone with `top_k=100` to retrieve a broad set of candidate chunks. These chunks are deduplicated by document identifier, retaining the top 10 unique documents ordered by relevance score.

**Step 3 -- Graph Expansion (GraphRAG)**

Starting from all 10 VDB-retrieved documents, the system queries Neo4j for related documents connected via CITES or HIGHER edges. Up to 5 neighbours are fetched per seed document. The combined set (VDB seeds + graph neighbours) is capped at 20 unique documents.

**Step 4 -- Alias-Aware Scoring**

Indonesian legal documents may be stored under different identifier conventions across systems. For example, "Permen PUPR 8/2022" may appear as `PERMENPUPR-NASIONAL-8-2022` in one system and `PERMEN-NASIONAL-8-2022` in another. The scoring engine builds a cross-system alias map that recognises these equivalences:

| Evidence Text         | Canonical Form                  | Known Alias                  |
|-----------------------|---------------------------------|------------------------------|
| Permen PUPR 8/2022    | PERMENPUPR-NASIONAL-8-2022      | PERMEN-NASIONAL-8-2022       |
| Permen PPN 7/2023     | PERMENPPN-NASIONAL-7-2023       | PERMEN-NASIONAL-7-2023       |
| Permen Perdagangan 24/2021 | PERMENDAG-NASIONAL-24-2021 | PERMEN-NASIONAL-24-2021      |
| Peraturan BPS 2/2020  | PERBANBPS-NASIONAL-2-2020       | PERMEN-NASIONAL-2-2020       |
| SK Dirjen BK 2022     | SKDIRJENBK-NASIONAL-12.1-2022   | KEPDIRJEN-NASIONAL-12.1-2022 |
| Pergub 20/2024        | PERGUB-PROVINSI-20-2024         | --                           |

A retrieved document matches a ground-truth document if either the canonical form or any of its aliases is present in the retrieved set.

### Configuration Parameters

| Parameter          | Value | Description                                           |
|--------------------|-------|-------------------------------------------------------|
| VDB_TOP_K          | 100   | Number of chunks retrieved from Pinecone              |
| VDB_MAX_DOCS       | 10    | Maximum unique documents retained from VDB            |
| GRAPHRAG_MAX_DOCS  | 20    | Maximum unique documents in GraphRAG result set       |
| NEO4J_NEIGHBOURS   | 5     | Graph neighbours fetched per seed document             |
| EXPAND_FROM_ALL    | True  | Expand from all VDB docs (not just top N)             |

### Metrics

- **Recall**: Fraction of ground-truth documents that appear in the retrieved set.
- **Precision**: Fraction of retrieved documents that match a ground-truth document.

Both metrics are computed per question and averaged across all scored questions.

---

## Results

### Dataset: QA 100 (All Sectors)

100 questions spanning construction, architecture, engineering, building, and workforce regulations.

| Metric                  | VDB-only | GraphRAG |
|-------------------------|----------|----------|
| Avg Recall              | 0.5560   | 0.8242   |
| Avg Precision           | 0.0800   | 0.0656   |

**Key observations:**
- GraphRAG achieves an average recall of 82.42%, a significant improvement over VDB-only at 55.60%.
- Graph expansion via Neo4j citation and hierarchical relationships recovers documents that semantic search alone misses.
- Precision is relatively low for both methods because the retrieved set (10 for VDB, up to 20 for GraphRAG) is intentionally broad to maximise recall. This is expected behaviour in a retrieval-first architecture where a downstream LLM performs the final synthesis.

### Dataset: QA Business 50

40 questions focused on corporate law, trade regulation, competition law, and investment policy.

| Metric                  | VDB-only | GraphRAG |
|-------------------------|----------|----------|
| Avg Recall              | 0.5250   | 0.6146   |
| Avg Precision           | 0.1225   | 0.0819   |

All 40 questions were scored successfully.

**Key observations:**
- GraphRAG improves average recall by 17% over VDB-only (0.61 vs. 0.53).
- The smaller improvement margin compared to QA 100 reflects the business dataset's reliance on regulations that are partially absent from Neo4j (e.g., UU 5/1999, UU 20/2008, UU 12/2011 are not indexed in the graph).
- Precision is lower for GraphRAG (0.08 vs. 0.12) because the larger result set (up to 20 documents) includes more non-relevant neighbours. GraphRAG prioritises recall at the cost of precision, which is the desired trade-off for a retrieval stage.

### Summary

| Dataset       | Questions | Recall VDB | Recall GraphRAG | Improvement |
|---------------|-----------|------------|-----------------|-------------|
| QA 100        | 100       | 0.5560     | 0.8242          | +48%        |
| QA Business   | 40        | 0.5250     | 0.6146          | +17%        |

---

## Output Files

### Detailed Retrieval Results

Location: `output/retrieval/detailed retrieval/`

Per-question CSV files with the following columns:

| Column                | Description                                                  |
|-----------------------|--------------------------------------------------------------|
| No                    | Question identifier                                          |
| Pertanyaan            | Question text (truncated to 200 characters)                  |
| GT_Total              | Number of ground-truth documents                             |
| GT_Doc_IDs            | Canonical ground-truth document identifiers                  |
| GT_From_Question      | Document identifiers extracted from question text            |
| Dok_VDB               | Documents retrieved by VDB-only                              |
| Dok_GraphRAG          | Documents retrieved by GraphRAG                              |
| Matched_GT_VDB        | Ground-truth documents matched by VDB results                |
| Matched_GT_GraphRAG   | Ground-truth documents matched by GraphRAG results           |
| Recall_VDB            | Recall score for VDB-only                                    |
| Precision_VDB         | Precision score for VDB-only                                 |
| Recall_GraphRAG       | Recall score for GraphRAG                                    |
| Precision_GraphRAG    | Precision score for GraphRAG                                 |

### Metrics Summary

Location: `output/retrieval/metrics/`

Aggregate CSV files containing:

| Metric                  | Description                                     |
|-------------------------|-------------------------------------------------|
| Total_Questions         | Total questions in the dataset                  |
| Scored_Questions        | Questions with valid retrieval results           |
| Skipped_Questions       | Questions skipped (no ground-truth)             |
| VDB_TOP_K               | Pinecone retrieval depth                        |
| VDB_MAX_DOCS            | Maximum unique VDB documents                    |
| GRAPHRAG_MAX_DOCS       | Maximum unique GraphRAG documents               |
| NEO4J_NEIGHBOURS        | Graph neighbours per seed                       |
| Avg_Recall_VDB          | Mean recall across scored questions (VDB)       |
| Avg_Precision_VDB       | Mean precision across scored questions (VDB)    |
| Avg_Recall_GraphRAG     | Mean recall across scored questions (GraphRAG)  |
| Avg_Precision_GraphRAG  | Mean precision across scored questions (GraphRAG)|
| Elapsed_Seconds         | Total execution time                            |

---

## Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

Required environment variables (in `.env`):

```
NEO4J_URI=neo4j+s://...
NEO4J_USERNAME=...
NEO4J_PASSWORD=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=...
HF_API_URL=...
HF_API_TOKEN=...
OPENROUTER_API_KEY=...
```

### Running the Benchmark

Process all XLSX files in the `benchmark/` directory:

```bash
python run_benchmark_v3.py
```

Process a single file:

```bash
python run_benchmark_v3.py benchmark/QA\ 100\ \(test-all-sector\).xlsx
```

Results are saved to:
- `output/retrieval/detailed retrieval/` -- per-question results
- `output/retrieval/metrics/` -- aggregate summaries

### Viewing Results in Streamlit

```bash
streamlit run app.py
```

Navigate to the "Compare Documents" tab to view benchmark results interactively.

### Running Kausalitas Analysis

```bash
python run_kausalitas.py
```

Results are saved to `output/kausalitas/`.

#### Classification Labels

Each document pair is classified using three labels based on Indonesian legal doctrine:

| Label | Meaning | Criteria |
|-------|---------|----------|
| **ENTAILMENT** | Regulations are aligned / complementary | Delegation/attribution relationship, substantive consistency, teleological alignment |
| **CONTRADICTION** | Regulations conflict / disharmony | Authority conflicts, contradictory obligations, terminology inconsistency, hierarchy violations (Lex Superior / Lex Specialis / Lex Posterior) |
| **NEUTRAL** | No substantive relationship | Separate jurisdictions, mutually exclusive subject matter, no normative overlap |

The LLM reasoning includes **specific Pasal (Article) and Ayat (Clause) citations** from each document to justify the classification.

#### Results Summary

27 unique document pairs analysed (from 53 Neo4j edges). Run time: ~55 seconds.

| Label | Count |
|-------|-------|
| **ENTAILMENT** | 14 |
| **NEUTRAL** | 13 |
| **CONTRADICTION** | 0 |
| **Error** | 0 |

#### Detailed Results

| # | Dokumen Sumber | Dokumen Pembanding | Relasi | Label | Alasan (ringkas) |
|---|---------------|-------------------|--------|-------|-----------------|
| 1 | KEPDIRJEN-NASIONAL-12.1-2022 | PP-NASIONAL-14-2021 | CITES, HIGHER | ENTAILMENT | Diktum KESATU angka 2 KEPDIRJEN mengoperasionalisasikan Pasal 168A PP 14/2021 tentang standar kompetensi kerja konstruksi. |
| 2 | KEPDIRJEN-NASIONAL-12.1-2022 | UU-NASIONAL-2-2017 | CITES, HIGHER | ENTAILMENT | Pasal 70 dan Pasal 95 UU 2/2017 mendelegasikan sertifikasi jabatan kerja konstruksi yang dioperasionalisasikan KEPDIRJEN. |
| 3 | PERDA_KAB-KABUPATEN-5-2015 | UU-NASIONAL-23-2014 | CITES, HIGHER | NEUTRAL | Konten PERDA tidak tersedia di VDB. |
| 4 | PERGUB-PROVINSI-135-2019 | UU-NASIONAL-23-2014 | CITES, HIGHER | NEUTRAL | PERGUB mengatur tata bangunan DKI Jakarta (Pasal 4-5); UU mengatur pemerintahan daerah (Pasal 213, 215, 218) — domain terpisah. |
| 5 | PERGUB-PROVINSI-135-2019 | UU-NASIONAL-28-2002 | CITES, HIGHER | ENTAILMENT | Pasal 7 ayat (2) UU 28/2002 mendelegasikan pembinaan bangunan gedung ke Pemda; Pergub 135/2019 mengoperasionalisasikannya. |
| 6 | PERGUB-PROVINSI-20-2024 | PP-NASIONAL-16-2021 | CITES, HIGHER | ENTAILMENT | PERGUB mengoperasionalisasikan PBG (Pasal 255, 259 PP 16/2021) dan standar teknis di tingkat daerah. |
| 7 | PERGUB-PROVINSI-20-2024 | UU-NASIONAL-23-2014 | CITES, HIGHER | NEUTRAL | PERGUB mengatur tata bangunan; UU mengatur organisasi DPRD provinsi — domain terpisah, tidak ada irisan. |
| 8 | PERGUB-PROVINSI-20-2024 | UU-NASIONAL-28-2002 | CITES, HIGHER | ENTAILMENT | Pasal 8, 9, 16 UU 28/2002 memberikan norma umum bangunan gedung; PERGUB mengatur teknis izin helipad, ramp, JBB. |
| 9 | PERMEN-NASIONAL-22-2018 | UU-NASIONAL-28-2002 | CITES, HIGHER | NEUTRAL | Konten PERMEN tidak tersedia di VDB. |
| 10 | PERMEN-NASIONAL-40-2022 | UU-NASIONAL-23-2014 | CITES, HIGHER | NEUTRAL | Konten PERMEN tidak tersedia di VDB. |
| 11 | PERMENDAG-NASIONAL-24-2021 | PP-NASIONAL-29-2021 | CITES | NEUTRAL | Konten PP 29/2021 tidak tersedia di VDB. |
| 12 | PERMENDAG-NASIONAL-24-2021 | UU-NASIONAL-11-2020 | CITES, HIGHER | NEUTRAL | PERMENDAG mengatur distribusi barang & keagenan; UU Cipta Kerja mengatur perizinan berusaha secara umum — domain terpisah. |
| 13 | PERMENKES-20-2022 | UU-NASIONAL-23-2014 | CITES, HIGHER | NEUTRAL | PERMENKES mengatur bangunan rumah sakit; UU mengatur organisasi DPRD dan perangkat daerah — domain terpisah. |
| 14 | PERMENPUPR-NASIONAL-8-2022 | UU-NASIONAL-11-2020 | CITES, HIGHER | NEUTRAL | PERMENPUPR mengatur sertifikasi konstruksi (Pasal 37-41); UU Cipta Kerja tidak mengatur substansi ini spesifik. |
| 15 | PERMENPUPR-NASIONAL-8-2022 | UU-NASIONAL-2-2017 | CITES, HIGHER | ENTAILMENT | Pasal 4-9, 42-45 UU 2/2017 didelegasikan ke PP; PERMENPUPR mengoperasionalisasikan via Pasal 37-41. |
| 16 | PERMENPUPR-NASIONAL-9-2020 | UU-NASIONAL-2-2017 | CITES, HIGHER | NEUTRAL | PERMENPUPR mengatur tata kelola internal LPJK (Pasal 20, 27, 29); UU mengatur sanksi dan sengketa konstruksi — domain terpisah. |
| 17 | PP-NASIONAL-14-2021 | UU-NASIONAL-11-2020 | CITES, HIGHER | ENTAILMENT | PP mengoperasionalisasikan sanksi administratif dan perizinan berusaha UU Cipta Kerja (Pasal 168A PP ↔ Pasal sanksi UU). |
| 18 | PP-NASIONAL-14-2021 | UU-NASIONAL-2-2017 | CITES, HIGHER | ENTAILMENT | Pasal 101 UU 2/2017 mendelegasikan sanksi administratif ke PP; dioperasionalisasikan Pasal 168A PP 14/2021. |
| 19 | PP-NASIONAL-15-2021 | UU-NASIONAL-11-2020 | CITES, HIGHER | NEUTRAL | PP mengatur praktik arsitek (Pasal 2, 9); UU Cipta Kerja tidak mengatur substansi ini — domain terpisah. |
| 20 | PP-NASIONAL-15-2021 | UU-NASIONAL-6-2017 | CITES, HIGHER | ENTAILMENT | Pasal 26-30, 32 UU 6/2017 mengatur Organisasi Profesi Arsitek; PP mengoperasionalisasikan tolok ukur kinerja. |
| 21 | PP-NASIONAL-16-2021 | UU-NASIONAL-11-2020 | CITES, HIGHER | ENTAILMENT | PP mengoperasionalisasikan PBG dan standar teknis (Pasal 232, 255, 259) yang didelegasikan UU Cipta Kerja. |
| 22 | PP-NASIONAL-16-2021 | UU-NASIONAL-28-2002 | CITES, HIGHER | ENTAILMENT | Pasal 44(5), 47(5), 48(3) UU 28/2002 mendelegasikan pengaturan teknis bangunan gedung ke PP 16/2021. |
| 23 | PP-NASIONAL-64-2016 | UU-NASIONAL-23-2014 | CITES, HIGHER | NEUTRAL | PP mengatur perizinan Perumahan MBR; UU mengatur organisasi pemerintahan daerah — domain terpisah. |
| 24 | PP-NASIONAL-64-2016 | UU-NASIONAL-28-2002 | CITES, HIGHER | ENTAILMENT | Pasal 7, 8, 42 UU 28/2002 mendelegasikan persyaratan dan perizinan bangunan; PP mengoperasionalisasikan via PTSP. |
| 25 | SK_DIRJEN_BK_2022 | PP-NASIONAL-14-2021 | CITES, HIGHER | ENTAILMENT | SK Dirjen merujuk Pasal 168A PP 14/2021 tentang standar kompetensi; mengoperasionalisasikan konversi jabatan kerja. |
| 26 | SK_DIRJEN_BK_2022 | UU-NASIONAL-2-2017 | CITES, HIGHER | ENTAILMENT | Pasal 70-75 UU 2/2017 tentang sertifikasi dan kualifikasi; SK Dirjen pelaksanaan teknis kodefikasi jabatan kerja. |
| 27 | UU-NASIONAL-40-2007 | UU-NASIONAL-23-2014 | CITES, HIGHER | NEUTRAL | UU 40/2007 mengatur Bantuan Hukum Pemda; UU 23/2014 mengatur organisasi DPRD — domain terpisah. |

#### Key Observations

- **No contradictions detected** across all 27 document pairs. This is expected because the regulations in the graph are primarily parent–child delegation chains (UU → PP → Permen → SK), where implementing regulations operationalise parent norms.
- **ENTAILMENT pairs** (14/27 = 52%) consistently show delegation/attribution relationships — higher-level legislation (UU) delegates specific matters to lower-level regulations (PP/Permen/SK) that operationalise them with technical detail.
- **NEUTRAL pairs** (13/27 = 48%) arise from two causes:
  - **Different domains** (9 pairs): The documents regulate entirely separate subject matters (e.g., building codes vs. regional government organisation).
  - **Missing VDB content** (4 pairs): Document content not available in Pinecone — classified as NEUTRAL by default.
- All ENTAILMENT reasonings include **specific Pasal/Ayat citations** from both documents (e.g., "Pasal 101 UU 2/2017 → Pasal 168A PP 14/2021"), enabling traceability of the legal delegation chain.

---

## Technical Stack

| Component       | Technology                                    |
|-----------------|-----------------------------------------------|
| Vector Database | Pinecone (1024-dim, index: lexport-trial)     |
| Graph Database  | Neo4j Aura (v5.27)                            |
| Embedding Model | Govnetic/Indo-LegalBERT-V3 (HuggingFace)     |
| LLM             | OpenAI GPT-4.1 (via OpenRouter)               |
| Web Interface   | Streamlit                                     |
| Runtime         | Python 3.10                                   |