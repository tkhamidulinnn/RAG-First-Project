from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import faiss
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Runtime stability for local environments (Jupyter / terminal).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


# -----------------------------------------------------------------------------
# Defaults (Week 2 notebook)
# -----------------------------------------------------------------------------
EMBEDDING_MODEL_ID = "hashing-768-stable"
EMBEDDING_DIM = 768
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K_DEFAULT = 3
RUNS_DIR = "runs"
# Week 4 notebook
SIMILARITY_THRESHOLD = 0.30
AMBIGUITY_GAP = 0.05
OLLAMA_MODEL = "gemma3:4b"
OLLAMA_TEMPERATURE = 0.2


# -----------------------------------------------------------------------------
# Guardrails (Week 4 notebook — same patterns, order, keys)
# -----------------------------------------------------------------------------
REFUSAL_REASONS = {
    "NO_CONTEXT": "No relevant context retrieved (similarity too low)",
    "AMBIGUOUS_RETRIEVAL": "Retrieval results are ambiguous (top-1 and top-2 too close)",
    "EMPTY_RETRIEVAL": "No documents retrieved",
    "PII_DETECTED": "Query asks for personally identifiable information",
    "PROMPT_INJECTION": "Potential prompt injection detected",
    "TOXICITY": "Query contains toxic or harmful content",
    "COMPETITOR_MENTION": "Query mentions restricted competitor or off-topic brand",
    "EMPTY_ANSWER": "Generated answer is empty",
    "UNCERTAIN_LANGUAGE": "Generated answer contains uncertain language",
    "DATA_LEAKAGE": "Response may leak internal system information",
}


def check_pii(query: str) -> bool:
    """Week 4: PII detection."""
    pii_keywords = [
        "email", "phone", "ssn", "social security",
        "credit card", "passport", "driver's license",
        "address", "zip code", "date of birth",
    ]
    return any(k in query.lower() for k in pii_keywords)


def check_prompt_injection(query: str) -> bool:
    """Week 4: prompt injection detection."""
    injection_patterns = [
        "ignore previous instructions", "ignore all instructions", "ignore the above",
        "disregard previous", "forget your instructions", "you are now", "act as",
        "pretend you are", "new instructions:", "system prompt:", "override:",
        "jailbreak", "do anything now", "developer mode",
    ]
    return any(p in query.lower() for p in injection_patterns)


def check_toxicity(query: str) -> bool:
    """Week 4: toxicity detection."""
    toxicity_keywords = [
        "how to hack", "how to steal", "how to kill",
        "make a bomb", "make a weapon", "illegal", "exploit vulnerability",
        "bypass security", "break into", "hate speech", "slur",
    ]
    return any(k in query.lower() for k in toxicity_keywords)


def check_competitor_mentions(query: str) -> bool:
    """Week 4: competitor mentions."""
    competitor_patterns = [
        "compare with chatgpt", "better than gpt", "switch to openai", "use openai instead",
        "compare with bard", "compare with gemini", "recommend competitor", "alternative product",
    ]
    return any(p in query.lower() for p in competitor_patterns)


def check_retrieval_quality(
    scores: list[float],
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    ambiguity_gap: float = AMBIGUITY_GAP,
) -> Tuple[bool, str | None]:
    """Week 4: retrieval quality gating."""
    if len(scores) == 0:
        return False, REFUSAL_REASONS["EMPTY_RETRIEVAL"]
    if scores[0] < similarity_threshold:
        return False, REFUSAL_REASONS["NO_CONTEXT"]
    if len(scores) >= 2 and (scores[0] - scores[1]) < ambiguity_gap:
        return False, REFUSAL_REASONS["AMBIGUOUS_RETRIEVAL"]
    return True, None


def check_data_leakage(answer: str) -> bool:
    """Week 4: data leakage check."""
    leakage_patterns = [
        "system prompt", "you are a helpful assistant",
        "api_key", "api key", "secret_key", "secret key",
        "access_token", "access token", "password:", "passwd:",
        "/etc/", "/var/", "c:\\\\", "internal use only", "confidential",
        "database connection", "connection string",
    ]
    return any(p in answer.lower() for p in leakage_patterns)


def check_generated_answer(answer: str) -> Tuple[bool, str | None]:
    """Week 4: post-check empty, leakage, uncertain language."""
    if not answer or not answer.strip():
        return False, REFUSAL_REASONS["EMPTY_ANSWER"]
    if "insufficient_context" in answer.lower():
        return True, None
    if check_data_leakage(answer):
        return False, REFUSAL_REASONS["DATA_LEAKAGE"]
    uncertain_phrases = [
        "i think", "i believe", "probably", "might", "possibly", "perhaps", "maybe", "could be",
        "i'm not sure", "i don't know", "uncertain",
    ]
    if any(p in answer.lower() for p in uncertain_phrases):
        return False, REFUSAL_REASONS["UNCERTAIN_LANGUAGE"]
    return True, None


def format_grounded_prompt_week4(context: str, query: str) -> str:
    """Week 4 create_grounded_prompt() — exact template from notebook."""
    return (
        "You are a helpful assistant that answers questions using ONLY the provided context.\n\n"
        "Context:\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        "Instructions:\n"
        "1. Answer the question using ONLY information from the context above.\n"
        "2. If the answer is not in the context, respond with exactly: \"INSUFFICIENT_CONTEXT\"\n"
        "3. Do not use any external knowledge or make guesses.\n"
        "4. If you are uncertain, respond with \"INSUFFICIENT_CONTEXT\"\n\n"
        "Answer:"
    )


# -----------------------------------------------------------------------------
# Data types
# -----------------------------------------------------------------------------
@dataclass
class DocumentChunk:
    chunk_id: int
    text: str
    source: str
    topic: str
    page: int


@dataclass
class RetrievalHit:
    rank: int
    score: float
    chunk: DocumentChunk


# -----------------------------------------------------------------------------
# LLM — Week 3 notebook: single implementation OllamaLLM gemma3:4b
# -----------------------------------------------------------------------------
class OllamaGemmaClient:
    """Week 3: OllamaLLM(model=gemma3:4b, temperature=0.2). Fails fast with instructions if unavailable."""

    def __init__(self) -> None:
        try:
            from langchain_ollama import OllamaLLM
            self._llm = OllamaLLM(model=OLLAMA_MODEL, temperature=OLLAMA_TEMPERATURE)
        except Exception as e:
            raise RuntimeError(
                "Ollama is required. Run: pip install langchain-ollama && ollama pull gemma3:4b && ollama serve. "
                "Then start this script again."
            ) from e

    def invoke(self, prompt: str) -> str:
        out = (self._llm.invoke(prompt) or "").strip()
        return out


# -----------------------------------------------------------------------------
# Monitor
# -----------------------------------------------------------------------------
class Monitor:
    def __init__(self, run_root: str = RUNS_DIR) -> None:
        self.run_id = uuid.uuid4().hex[:10]
        self.run_dir = Path(run_root) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.timings: dict[str, float] = {}

    def timeit(self, name: str):
        class _Timer:
            def __init__(self, outer: "Monitor", metric_name: str):
                self.outer = outer
                self.metric_name = metric_name
                self.start = 0.0

            def __enter__(self):
                self.start = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc, tb):
                self.outer.timings[self.metric_name] = round((time.perf_counter() - self.start) * 1000.0, 3)

        return _Timer(self, name)

    def write_json(self, filename: str, payload: dict[str, Any]) -> Path:
        path = self.run_dir / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


# -----------------------------------------------------------------------------
# VectorDatabase (Week 2: load by topic, chunk 300/50, HashingVectorizer 768, FAISS IndexFlatIP)
# -----------------------------------------------------------------------------
class VectorDatabase:
    def __init__(
        self,
        dim: int = EMBEDDING_DIM,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        monitor: Monitor | None = None,
    ) -> None:
        self.dim = dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.monitor = monitor or Monitor()

        self.vectorizer = HashingVectorizer(n_features=self.dim, alternate_sign=False, norm="l2")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.chunks: list[DocumentChunk] = []
        self.index: faiss.IndexFlatIP | None = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        x = self.vectorizer.transform(texts).astype(np.float32).toarray()
        faiss.normalize_L2(x)
        return np.ascontiguousarray(x)

    def _load_pages(self, data_path: str) -> list[dict[str, Any]]:
        root = Path(data_path)
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Data path not found: {data_path}")

        pages: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []

        for topic_dir in sorted(root.iterdir()):
            if not topic_dir.is_dir():
                continue
            topic = topic_dir.name

            for pdf_path in sorted(topic_dir.glob("*.pdf")):
                try:
                    for page in PyPDFLoader(str(pdf_path)).load():
                        pages.append(
                            {
                                "text": page.page_content,
                                "source": pdf_path.name,
                                "topic": topic,
                                "page": int(page.metadata.get("page", 0)),
                            }
                        )
                except Exception as e:
                    errors.append({"source": pdf_path.name, "error": str(e)})

            for txt_path in sorted(topic_dir.glob("*.txt")) + sorted(topic_dir.glob("*.md")):
                try:
                    txt = txt_path.read_text(encoding="utf-8", errors="replace")
                    pages.append(
                        {
                            "text": txt,
                            "source": txt_path.name,
                            "topic": topic,
                            "page": 0,
                        }
                    )
                except Exception as e:
                    errors.append({"source": txt_path.name, "error": str(e)})

        if errors:
            self.monitor.write_json("11_ingest_errors.json", {"errors": errors[:100], "n_errors": len(errors)})

        return pages

    def update_database(self, data_path: str) -> None:
        with self.monitor.timeit("ingest_ms"):
            pages = self._load_pages(data_path)

        chunks: list[DocumentChunk] = []
        chunk_id = 0

        with self.monitor.timeit("chunk_ms"):
            for p in pages:
                for part in self.splitter.split_text(p["text"]):
                    if not part.strip():
                        continue
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            text=part,
                            source=p["source"],
                            topic=p["topic"],
                            page=p["page"],
                        )
                    )
                    chunk_id += 1

        if not chunks:
            raise RuntimeError("No chunks produced from input data")

        with self.monitor.timeit("embed_ms"):
            vectors = self._embed([c.text for c in chunks])

        with self.monitor.timeit("index_ms"):
            idx = faiss.IndexFlatIP(vectors.shape[1])
            idx.add(vectors)
            self.index = idx
            self.chunks = chunks

        self.monitor.write_json(
            "00_config.json",
            {
                "embedding_model": EMBEDDING_MODEL_ID,
                "embedding_dim": self.dim,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "top_k_default": TOP_K_DEFAULT,
            },
        )
        self.monitor.write_json(
            "10_ingest_summary.json",
            {
                "data_path": data_path,
                "n_pages": len(pages),
                "n_chunks": len(chunks),
                "ntotal": int(self.index.ntotal if self.index is not None else 0),
            },
        )
        self.monitor.write_json("90_timings.json", self.monitor.timings)

    def Update(self, data_path: str) -> None:  # noqa: N802
        self.update_database(data_path)

    def UpdateDatabase(self, data_path: str) -> None:  # noqa: N802
        self.update_database(data_path)

    def recover(self, query: str, top_k: int = TOP_K_DEFAULT) -> list[tuple[DocumentChunk, dict[str, Any], float]]:
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Index is empty. Call update_database(path) first.")

        with self.monitor.timeit("query_embed_ms"):
            qv = self._embed([query])

        with self.monitor.timeit("search_ms"):
            scores, indices = self.index.search(qv, min(top_k, self.index.ntotal))

        out: list[tuple[DocumentChunk, dict[str, Any], float]] = []
        debug = []
        for rank, (s, i) in enumerate(zip(scores[0], indices[0]), 1):
            c = self.chunks[int(i)]
            ref = {"source": c.source, "topic": c.topic, "page": c.page, "chunk_id": c.chunk_id}
            score = float(s)
            out.append((c, ref, score))
            debug.append(
                {
                    "rank": rank,
                    "score": round(score, 4),
                    "source": c.source,
                    "topic": c.topic,
                    "page": c.page,
                    "chunk_id": c.chunk_id,
                    "text_preview": c.text[:180].replace("\n", " "),
                }
            )

        self.monitor.write_json(
            "30_retrieval_debug.json",
            {"query": query, "top_k": top_k, "results": debug, "timings": self.monitor.timings},
        )
        self.monitor.write_json("90_timings.json", self.monitor.timings)
        return out

    def Recover(self, query: str, top_k: int = TOP_K_DEFAULT):  # noqa: N802
        return self.recover(query, top_k=top_k)


# -----------------------------------------------------------------------------
# RAG (Week 4 generate_with_guardrails: PRE → LLM → POST → Sources)
# -----------------------------------------------------------------------------
def _citation(ref: dict[str, Any]) -> str:
    return f"topic={ref['topic']} source={ref['source']} page={ref['page']} chunk_id={ref['chunk_id']}"


def _format_context_week3(chunks: list[tuple[Any, dict[str, Any], float]]) -> str:
    """Week 3: format_context — [Chunk i] (score=...) + text."""
    parts = []
    for rank, (chunk, ref, score) in enumerate(chunks, 1):
        parts.append(f"[Chunk {rank}] (score={score:.4f}) {_citation(ref)}\n{chunk.text}")
    return "\n\n".join(parts)


class RAG:
    def __init__(
        self,
        vector_db: VectorDatabase,
        llm: OllamaGemmaClient | None = None,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        ambiguity_gap: float = AMBIGUITY_GAP,
    ) -> None:
        self.vector_db = vector_db
        self.llm = llm
        self.similarity_threshold = similarity_threshold
        self.ambiguity_gap = ambiguity_gap

    def retrieve(self, query: str, top_k: int = TOP_K_DEFAULT) -> str:
        """Week 4: PRE-checks (injection, toxicity, pii, competitor, retrieval_quality) → context → prompt → LLM → POST (INSUFFICIENT_CONTEXT, check_generated_answer) → Sources."""
        if self.llm is None:
            raise RuntimeError("LLM required for retrieve(). Use interactive mode or pass OllamaGemmaClient().")
        hits = self.vector_db.recover(query, top_k=top_k)
        scores: list[float] = [s for _, _, s in hits]
        citations = [_citation(ref) for _, ref, _ in hits]
        sources_block = "Sources:\n" + "\n".join(f"  [{i+1}] {c}" for i, c in enumerate(citations))

        def refuse(reason: str, stage: str, raw: str | None = None) -> str:
            msg = f"Blocked ({stage}): {reason}\n\n{sources_block}"
            self._write_generation_debug(query, "refused", stage, reason, scores, 0, OLLAMA_MODEL, raw)
            return msg

        if not hits:
            return refuse(REFUSAL_REASONS["EMPTY_RETRIEVAL"], "pre")

        # PRE 1–5 (Week 4 order)
        if check_prompt_injection(query):
            return refuse(REFUSAL_REASONS["PROMPT_INJECTION"], "pre")
        if check_toxicity(query):
            return refuse(REFUSAL_REASONS["TOXICITY"], "pre")
        if check_pii(query):
            return refuse(REFUSAL_REASONS["PII_DETECTED"], "pre")
        if check_competitor_mentions(query):
            return refuse(REFUSAL_REASONS["COMPETITOR_MENTION"], "pre")
        retrieval_ok, retrieval_reason = check_retrieval_quality(
            scores, self.similarity_threshold, self.ambiguity_gap
        )
        if not retrieval_ok and retrieval_reason:
            return refuse(retrieval_reason, "pre")

        context = _format_context_week3(hits)
        prompt = format_grounded_prompt_week4(context, query)

        try:
            raw_answer = self.llm.invoke(prompt)
        except Exception as e:
            return refuse(f"LLM error: {str(e)[:100]}", "post")

        # POST: INSUFFICIENT_CONTEXT (Week 4)
        if raw_answer and "insufficient_context" in raw_answer.lower():
            out = "INSUFFICIENT_CONTEXT\n\n" + sources_block
            self._write_generation_debug(query, "insufficient_context", "final", "INSUFFICIENT_CONTEXT", scores, len(prompt), OLLAMA_MODEL, raw_answer[:2000])
            return out

        # POST: check_generated_answer (Week 4)
        answer_ok, answer_reason = check_generated_answer(raw_answer)
        if not answer_ok and answer_reason:
            return refuse(answer_reason, "post", raw_answer[:2000] if raw_answer else None)

        answer_with_sources = raw_answer.rstrip() + "\n\n" + sources_block
        self._write_generation_debug(query, "answer", "final", None, scores, len(prompt), OLLAMA_MODEL, raw_answer[:2000])
        return answer_with_sources

    def _write_generation_debug(
        self,
        query: str,
        outcome: str,
        stage: str,
        reason: str | None,
        scores: list[float],
        prompt_length: int,
        model_used: str,
        raw_answer: str | None,
    ) -> None:
        """40_generation_debug.json: query, retrieval scores, guardrail decision, model."""
        payload = {
            "query": query,
            "outcome": outcome,
            "stage": stage,
            "reason": reason,
            "max_score": round(scores[0], 4) if scores else None,
            "score_gap": round(scores[0] - scores[1], 4) if len(scores) >= 2 else None,
            "prompt_length": prompt_length,
            "model_used": model_used,
            "raw_answer": (raw_answer or "")[:2000],
        }
        self.vector_db.monitor.write_json("40_generation_debug.json", payload)

    def Retrieve(self, query: str, top_k: int = TOP_K_DEFAULT) -> str:  # noqa: N802
        return self.retrieve(query, top_k=top_k)


class Vectordatabase(VectorDatabase):
    pass


class Rag(RAG):
    pass


# -----------------------------------------------------------------------------
# Evaluation (Week 5: expected_topic JSONL, hit@k)
# -----------------------------------------------------------------------------
def hit_at_k(expected_topic: str | None, retrieved_topics: list[str], k: int) -> float | None:
    if expected_topic is None:
        return None
    return 1.0 if expected_topic in retrieved_topics[:k] else 0.0


def run_eval(rag: RAG, eval_jsonl: str, top_k: int) -> None:
    path = Path(eval_jsonl)
    if not path.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_jsonl}")

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        query = row.get("query") or row.get("question")
        expected_topic = row.get("expected_topic")

        hits = rag.vector_db.recover(query, top_k=top_k)
        topics = [ref["topic"] for _, ref, _ in hits]

        rows.append(
            {
                "query": query,
                "expected_topic": expected_topic,
                "retrieved_topics": topics,
                "hit@1": hit_at_k(expected_topic, topics, 1),
                "hit@3": hit_at_k(expected_topic, topics, 3),
                "hit@5": hit_at_k(expected_topic, topics, 5),
            }
        )

    with_expected = [r for r in rows if r["expected_topic"] is not None]
    report: dict[str, Any] = {"n_queries": len(rows), "rows": rows}
    if with_expected:
        report["hit@1_mean"] = sum(r["hit@1"] for r in with_expected) / len(with_expected)
        report["hit@3_mean"] = sum(r["hit@3"] for r in with_expected) / len(with_expected)
        report["hit@5_mean"] = sum(r["hit@5"] for r in with_expected) / len(with_expected)

    rag.vector_db.monitor.write_json("60_eval_report.json", report)
    print("Eval report written:", rag.vector_db.monitor.run_dir / "60_eval_report.json")


# -----------------------------------------------------------------------------
# CLI (only flags used in notebooks: data_dir, top_k, dim, interactive, eval)
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="RAG pipeline (notebooks Week 2–5)")
    parser.add_argument("--data_dir", default="data", help="Data directory with topic subfolders")
    parser.add_argument("--top_k", type=int, default=TOP_K_DEFAULT)
    parser.add_argument("--dim", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--eval", default="", help="Path to JSONL (query, expected_topic)")
    args = parser.parse_args()

    if not args.interactive and not args.eval:
        parser.error("Specify either --interactive or --eval <path>")

    monitor = Monitor()
    vd = VectorDatabase(dim=args.dim, monitor=monitor)
    vd.update_database(args.data_dir)

    if args.interactive:
        llm = OllamaGemmaClient()
        rag = RAG(vd, llm=llm)
        print("Interactive mode. Empty query exits.")
        while True:
            q = input("Query> ").strip()
            if not q:
                break
            print(rag.retrieve(q, top_k=args.top_k))
            print()
    else:
        rag = RAG(vd, llm=None)
        run_eval(rag, args.eval, args.top_k)


if __name__ == "__main__":
    main()
