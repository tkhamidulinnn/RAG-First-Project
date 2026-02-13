# Week 3 — LLM Integration & Prompt Engineering: Reference

## 1) Position in the project

Week 3 is built directly on Week 1 + Week 2 outputs:

- Week 1: quality benchmark selected **MPNet + cosine** as the preferred semantic setup.
- Week 2: runtime execution in this environment switched to **stable hashing embedder + FAISS** due to native kernel crashes on `sentence-transformers/torch` under Python 3.13 Jupyter.
- Week 3: keeps the **Week 2 runtime retriever** unchanged and adds prompt + local LLM generation.

Pipeline:

`question -> retrieve chunks -> build prompt -> generate answer -> log metrics`

## 2) Why Week 3 uses Week 2 runtime retriever

Week 3 depends on a retriever that is reproducible in local notebook execution. In this repository, the stable option is:

- embedder: `hashing-768-stable` (HashingVectorizer-based)
- index: FAISS `IndexFlatIP`
- chunking: `chunk_size=300`, `chunk_overlap=50`

This is an engineering decision for reliability. It does not invalidate Week 1 benchmark results.

## 3) Key files

- `notebooks/week3_llm_integration_and_prompts.ipynb`: end-to-end Week 3 notebook.
- `source/week3_llm_client.py`: LLM call wrapper.
- `source/week3_prompt_templates.py`: prompt templates (`qa_strict`, `summary`, `reasoning`).
- `source/week3_rag_chain.py`: orchestration helpers.
- `artifacts/week3_prompt_experiments.csv`: experiment table.
- `artifacts/week3_generations.jsonl`: generation logs.
- `artifacts/week3_summary.md`: final summary.

## 4) Core design choices

### Retrieval

- Reuse Week 2 retriever interface (`VectorStore`, `RAG`) to keep continuity.
- Keep cosine-equivalent FAISS scoring (`IndexFlatIP` on L2-normalized hashing vectors).
- Keep fixed chunking to isolate prompt/LLM effects from retrieval changes.

### Prompting

- `qa_strict`: strict grounding, explicit fallback when context is insufficient.
- `summary`: concise synthesis mode.
- `reasoning`: structured explanation mode.

### LLM

- Local Ollama model for reproducibility and offline development.
- Temperature grid (`0.0`, `0.2`) for controlled behavior comparison.

## 5) Metrics and interpretation

Main columns used in Week 3 analysis:

- `latency_s`: generation latency.
- `answer_len`: answer length proxy.
- `idk_flag`: whether model refused due to missing evidence.
- `has_citation_flag`: whether answer includes chunk citations.

Interpretation:

- Higher citation rate + lower hallucination behavior is preferred for RAG.
- Temperature mainly affects style; retrieval quality remains the dominant factor.

## 6) Practical guidance

If you rerun Week 3:

1. Ensure Week 2 runtime assumptions remain unchanged (same chunking and retriever).
2. Run retrieval setup first, then prompt experiments.
3. Compare templates primarily on grounding behavior, not only readability.

## 7) Connection to Week 4

Week 4 adds guardrails around this Week 3 chain:

- input filtering,
- retrieval quality checks,
- output validation.

That keeps the same architecture while adding control and safety.
