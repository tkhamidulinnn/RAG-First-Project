# Week 2 — Local Retrieval Pipeline: Reference

## 1) Goal

Build a local retrieval pipeline over project PDFs:

`PDF -> chunk -> embed -> FAISS index -> top-k retrieval`

Week 2 output is consumed by Week 3 as the retrieval layer.

## 2) Inputs from Week 1

Week 1 benchmark conclusion:

- preferred semantic model: **MPNet**
- preferred similarity metric: **cosine**

## 3) Why Week 2 runtime differs from Week 1 model decision

In this local environment (`Python 3.13`, macOS ARM, Jupyter), the MPNet path (`sentence-transformers/torch`) repeatedly caused **native kernel termination** during indexing.

Observed behavior:

- Jupyter session dies with no Python traceback.
- Kernel restarts; execution is non-reproducible.

Engineering decision for Week 2 runtime:

- switch embedder to **`hashing-768-stable`** (HashingVectorizer-based),
- keep FAISS and cosine-equivalent ranking,
- preserve pipeline API and evaluation flow.

This is a stability fallback, not a claim that hashing is semantically better than MPNet.

## 4) Final Week 2 runtime configuration

- Chunking: `chunk_size=300`, `chunk_overlap=50`
- Embedder: `hashing-768-stable`
- Index: FAISS `IndexFlatIP`
- Similarity: cosine-equivalent (L2-normalized vectors + inner product)

## 5) Main components

### `load_pdfs(data_dir)`

- Loads PDF pages from `data/<topic>/*.pdf`
- Adds metadata: `topic`, `source`
- Skips unreadable PDFs with explicit error print

### `chunk_documents(...)`

- Uses `RecursiveCharacterTextSplitter`
- Fixed parameters to keep downstream experiments comparable

### `VectorStore`

- Builds FAISS index in streaming batches
- Stores chunks and returns top-k `(scores, indices)`

### `RAG`

- Wraps `VectorStore` retrieval
- Returns `(score, chunk)` pairs for evaluation and prompt context

## 6) Evaluation artifacts

Week 2 writes:

- `artifacts/week2_retrieval_eval.csv`
- `artifacts/week2_retrieval_summary.csv`
- `artifacts/week2_final_decisions.csv`

These files are the handoff contract for Week 3.

## 7) Trade-offs (explicit)

Advantages of fallback in Week 2:

- stable notebook execution,
- reproducible local runs,
- no kernel death during indexing.

Cost:

- weaker semantic embeddings versus MPNet in general.

Rationale:

- for Week 2 objective (pipeline assembly + retrieval evaluation), reproducibility is mandatory.

## 8) How to explain this to mentor/reviewer

Short version:

1. Week 1 selected MPNet by quality.
2. Week 2 runtime with MPNet crashed at native level in this environment.
3. We switched to a stable embedder to complete retrieval pipeline goals reliably.
4. Architecture and evaluation discipline were preserved; only embedding backend changed.
