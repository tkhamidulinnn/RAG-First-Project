# RAG Pipeline

Local, reproducible RAG system built over a 6-week internship.
Implementation: [`tragframe.py`](tragframe.py) — Architecture spec: [`RAG.ipynb`](RAG.ipynb)

---

## Quick start

```bash
# Interactive — ask questions in the terminal
python tragframe.py --interactive --data_dir data

# Evaluation — batch Hit@k from a JSONL file
python tragframe.py --eval eval.jsonl --data_dir data
```

---

## How the pipeline works

```
data/<topic>/*.pdf
        │
        ▼  ingest (PyPDFLoader)
   raw pages + topic label
        │
        ▼  chunk (300 chars / 50 overlap)
   DocumentChunk list
        │
        ▼  embed (HashingVectorizer 768-dim)
   float32 vectors
        │
        ▼  index (FAISS IndexFlatIP)
   vector store
        │
   ─────┼───────────────────── query time ──
        │
   user query
        │
        ▼  guardrails (PII / injection / toxicity / quality gate)
        │
        ▼  VectorDatabase.recover(query, top_k)
   top-k chunks + scores + citations
        │
        ▼  RAG.retrieve()  →  OllamaLLM (gemma3:4b)
   grounded answer
        │
        ▼  Monitor
   runs/<run_id>/  artifacts
```

---

## Setup

```bash
python -m venv rag_env
source rag_env/bin/activate       
pip install -r requirements.txt
```

Ollama is required for LLM generation:

```bash
ollama pull gemma3:4b
ollama serve
```

---

## Data layout

Place documents under `data/<topic>/`:

```
data/
  RAG/
    paper.pdf
  GIT/
    guide.pdf
    notes.txt
  GCP/
    intro.pdf
    overview.md
```

Supported formats: `.pdf` `.txt` `.md`
Topic label = subfolder name (e.g. `GIT`, `RAG`, `GCP`).

---

## Run modes

### Interactive — ask questions in the terminal

```bash
python tragframe.py --interactive --data_dir data
```

```
Query> What is RAG?
Query> How do I create a git branch?
Query>          ← empty line to exit
```

### Evaluation — batch metrics from a JSONL file

Create `eval.jsonl`:

```json
{"query": "What is RAG?",                      "expected_topic": "RAG"}
{"query": "How to create a git branch?",       "expected_topic": "GIT"}
{"query": "What is Google Cloud Platform?",    "expected_topic": "GCP"}
{"query": "What is the capital of France?",    "expected_topic": null}
```

Run:

```bash
python tragframe.py --eval eval.jsonl --data_dir data
```

Outputs Hit@1 / Hit@3 / Hit@5 per query and macro-averages.

---

## Output artifacts

Every run writes to `runs/<run_id>/`:

| File | Contents |
|------|----------|
| `00_config.json` | Embedding dim, chunk size, top_k |
| `10_ingest_summary.json` | Pages loaded, chunks created |
| `11_ingest_errors.json` | Files that failed to load (if any) |
| `30_retrieval_debug.json` | Query, top-k hits, scores, citations |
| `40_generation_debug.json` | Guardrail outcome, model used, raw answer |
| `60_eval_report.json` | Hit@k per query + macro averages (eval mode only) |
| `90_timings.json` | ms per stage: ingest, chunk, embed, index, search |

Each run has a unique `run_id` — artifacts never overwrite each other.

---

## Key design decisions

| Decision | Reason |
|----------|--------|
| `HashingVectorizer` instead of transformers | No OOM risk; deterministic; zero cold-start |
| `FAISS IndexFlatIP` | Inner product; no training required |
| Guardrails before and after LLM | Block PII, injection, toxicity, ambiguous retrieval, data leakage |
| Artifacts per run | Full auditability; every past run is inspectable |
| Topic label from folder name | Deterministic; no NLP required at ingest time |

---
