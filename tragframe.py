"""RAG framework: Monitor, VectorDatabase, RAG. Import-safe — executes nothing on load."""
from __future__ import annotations

import json
import os
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import faiss
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
EMBEDDING_MODEL_ID = "hashing-768-stable"
EMBEDDING_DIM = 768
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K_DEFAULT = 3
RUNS_DIR = "runs"
SIMILARITY_THRESHOLD = 0.20
AMBIGUITY_GAP = 0.0

REFUSAL_MESSAGE = "This query cannot be processed."
REFUSAL_REASONS: dict[str, str] = {
    "PROMPT_INJECTION": "Prompt injection detected.",
    "TOXICITY": "Toxic content detected.",
    "PII_DETECTED": "Personal information detected.",
    "COMPETITOR_MENTION": "Competitor mention detected.",
    "EMPTY_RETRIEVAL": "No documents retrieved.",
    "NO_CONTEXT": "Retrieved context has insufficient relevance.",
    "AMBIGUOUS_RETRIEVAL": "Retrieval results are ambiguous.",
    "EMPTY_ANSWER": "Generated answer is empty.",
    "DATA_LEAKAGE": "Response contains sensitive data.",
    "UNCERTAIN_LANGUAGE": "Response expresses uncertainty instead of grounded facts.",
}


# -----------------------------------------------------------------------------
# Guardrail functions
# -----------------------------------------------------------------------------
_PII_KEYWORDS: list[str] = [
    "email", "phone", "ssn", "social security",
    "credit card", "passport", "driver's license",
    "address", "zip code", "date of birth",
]
_INJECTION_PATTERNS: list[str] = [
    "ignore previous instructions", "ignore all instructions", "ignore the above",
    "disregard previous", "forget your instructions", "you are now", "act as",
    "pretend you are", "new instructions:", "system prompt:", "override:",
    "jailbreak", "do anything now", "developer mode",
]
_TOXICITY_KEYWORDS: list[str] = [
    "how to hack", "how to steal", "how to kill",
    "make a bomb", "make a weapon", "illegal", "exploit vulnerability",
    "bypass security", "break into", "hate speech", "slur",
]
_COMPETITOR_PATTERNS: list[str] = [
    "compare with chatgpt", "better than gpt", "switch to openai", "use openai instead",
    "compare with bard", "compare with gemini", "recommend competitor", "alternative product",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    """Checks whether any pattern from the list appears in the lowercased text. Shared helper used by all keyword-based guardrail checks."""
    lower = text.lower()
    return any(p in lower for p in patterns)


# Input guardrails — applied to the raw query before retrieval/generation.
def check_pii(query: str) -> bool:
    """Returns True if the query contains personal identifiable information keywords. Blocks queries that ask about email, phone, SSN, credit card, and similar data."""
    return _matches_any(query, _PII_KEYWORDS)


def check_prompt_injection(query: str) -> bool:
    """Returns True if the query contains prompt injection or jailbreak patterns. Blocks attempts to override system instructions or manipulate the model's behaviour."""
    return _matches_any(query, _INJECTION_PATTERNS)


def check_toxicity(query: str) -> bool:
    """Returns True if the query contains harmful or illegal request keywords. Blocks queries about hacking, weapons, hate speech, and similar content."""
    return _matches_any(query, _TOXICITY_KEYWORDS)


def check_competitor_mentions(query: str) -> bool:
    """Returns True if the query asks to compare with or switch to a competitor product. Blocks competitor comparison and recommendation requests."""
    return _matches_any(query, _COMPETITOR_PATTERNS)


def check_retrieval_quality(
    scores: list[float],
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    ambiguity_gap: float = AMBIGUITY_GAP,
) -> Tuple[bool, str | None]:
    """Validates retrieval scores before passing context to the LLM. Returns (True, None) on pass, or (False, reason) if scores are empty, below the similarity threshold, or ambiguous."""
    if not scores:
        return False, REFUSAL_REASONS["EMPTY_RETRIEVAL"]
    if scores[0] < similarity_threshold:
        return False, REFUSAL_REASONS["NO_CONTEXT"]
    if len(scores) >= 2 and (scores[0] - scores[1]) < ambiguity_gap:
        return False, REFUSAL_REASONS["AMBIGUOUS_RETRIEVAL"]
    return True, None


# Output guardrail — applied to the generated answer before returning to the caller.
def check_data_leakage(answer: str) -> bool:
    """Returns True if the generated answer contains sensitive system or credential patterns. Blocks responses that accidentally expose API keys, system prompts, or internal paths."""
    # Patterns that leak system/config (aligned with week4 experiments).
    leakage_patterns = [
        "system prompt",
        "you are a helpful assistant",
        "api_key", "api key", "secret_key", "secret key",
        "access_token", "access token", "password:", "passwd:",
        "/etc/", "/var/", "c:\\\\", "internal use only", "confidential",
        "database connection", "connection string",
    ]
    return any(p in answer.lower() for p in leakage_patterns)


# Phrases that indicate the model is refusing or guessing rather than answering from context.
# Aligned with week4 experiments.
UNCERTAIN_PHRASES = (
    "i think", "i believe", "probably", "might",
    "possibly", "perhaps", "maybe", "could be",
    "i'm not sure", "i don't know", "uncertain",
)


def check_generated_answer(answer: str) -> Tuple[bool, str | None]:
    """Validates the generated answer for emptiness, data leakage, and uncertain language. Returns (True, None) on pass or (False, reason) on failure."""
    if not answer or not answer.strip():
        return False, REFUSAL_REASONS["EMPTY_ANSWER"]
    if "insufficient_context" in answer.lower():
        return True, None
    if check_data_leakage(answer):
        return False, REFUSAL_REASONS["DATA_LEAKAGE"]
    answer_lower = answer.lower()
    if any(p in answer_lower for p in UNCERTAIN_PHRASES):
        return False, REFUSAL_REASONS["UNCERTAIN_LANGUAGE"]
    return True, None


def _pre_guardrails_block(query: str) -> Tuple[bool, str | None]:
    """Returns (True, None) if query passes all pre-generation checks, else (False, reason)."""
    if check_prompt_injection(query):
        return False, REFUSAL_REASONS["PROMPT_INJECTION"]
    if check_toxicity(query):
        return False, REFUSAL_REASONS["TOXICITY"]
    if check_pii(query):
        return False, REFUSAL_REASONS["PII_DETECTED"]
    if check_competitor_mentions(query):
        return False, REFUSAL_REASONS["COMPETITOR_MENTION"]
    return True, None


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _citation(ref: dict[str, Any]) -> str:
    """Formats a chunk reference dict into a single-line citation string. Used to label retrieved chunks in the sources block appended to every answer."""
    return f"topic={ref['topic']} source={ref['source']} page={ref['page']} chunk_id={ref['chunk_id']}"


def _format_context(chunks: list[tuple[Any, dict[str, Any], float]]) -> str:
    """Formats a list of retrieved chunks into a numbered context block for the LLM prompt. Each entry is prefixed with its rank, score, and citation."""
    return "\n\n".join(
        f"[Chunk {rank}] (score={score:.4f}) {_citation(ref)}\n{chunk.text}"
        for rank, (chunk, ref, score) in enumerate(chunks, 1)
    )


def _eval_hit_correct(
    chunk: "DocumentChunk",
    ref: dict[str, Any],
    gold_chunk_ids: list[int],
    gold_sources: list[str],
    gold_keywords: list[str],
) -> bool:
    """Returns True if the retrieved chunk matches any of the provided gold criteria. Checks chunk_id, source substring, and keyword substring in that order."""
    if gold_chunk_ids and chunk.chunk_id in gold_chunk_ids:
        return True
    if gold_sources and any(gs.lower() in ref["source"].lower() for gs in gold_sources):
        return True
    if gold_keywords and any(kw.lower() in chunk.text.lower() for kw in gold_keywords):
        return True
    return False


def _hit_mrr_scores(correct_ranks: list[int]) -> Tuple[float, float, float, float]:
    """Returns (hit@1, hit@3, hit@5, mrr@5) for the given list of correct ranks (ascending)."""
    best = correct_ranks[0] if correct_ranks else None
    hit1 = 1.0 if best == 1 else 0.0
    hit3 = 1.0 if best is not None and best <= 3 else 0.0
    hit5 = 1.0 if best is not None else 0.0
    mrr = 1.0 / best if best else 0.0
    return (hit1, hit3, hit5, mrr)


def _mean(lst: list[float]) -> float:
    """Returns the rounded mean of a float list, or 0.0 for an empty list."""
    return round(sum(lst) / len(lst), 4) if lst else 0.0


def _grounded_prompt(context: str, query: str) -> str:
    """Builds the system prompt that instructs the LLM to answer using only the retrieved context. Requires chunk citations and mandates INSUFFICIENT_CONTEXT when evidence is missing."""
    return (
        "You are a helpful assistant that answers questions using ONLY the provided context.\n\n"
        "Context:\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        "Instructions:\n"
        "1. Answer using ONLY information from the context above. Base every fact on the context; cite [Chunk N] where N is the chunk number when applicable.\n"
        "2. If the answer is not in the context or you are uncertain, respond with exactly: \"INSUFFICIENT_CONTEXT\".\n"
        "3. Do not use any external knowledge or make guesses. There is no limit on answer length — include all relevant information from the context.\n\n"
        "Answer:"
    )


# -----------------------------------------------------------------------------
# Junk filter
# -----------------------------------------------------------------------------

_NAV_KEYWORDS = {
    "products", "pricing", "blog", "careers", "company", "about", "partners",
    "resources", "community", "documentation", "status", "integrations",
    "learning center", "customer case studies", "on this page", "powered by",
    "try", "ask ai",
}


def _is_junk_chunk(text: str) -> bool:
    """Returns True if the chunk looks like navigation, menu, or TOC noise rather than document content. Filters out very short chunks, nav-bar keyword clusters, and chunks with very short average line length."""
    s = " ".join(text.split())  # normalize whitespace
    if len(s) < 80:
        return True
    if sum(kw in s.lower() for kw in _NAV_KEYWORDS) >= 3 and s.count(".") + s.count("?") + s.count("!") == 0:
        return True
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return len(lines) > 8 and sum(len(ln) for ln in lines) / len(lines) < 20


# -----------------------------------------------------------------------------
# Data types
# -----------------------------------------------------------------------------
@dataclass
class DocumentChunk:
    """Holds a single text chunk and its provenance metadata after splitting a PDF page."""
    chunk_id: int
    text: str
    source: str
    topic: str
    page: int


# -----------------------------------------------------------------------------
# Monitor
# -----------------------------------------------------------------------------
class Monitor:
    """Generates a unique run directory and collects latency timings and JSON artifacts."""

    def __init__(self, run_root: str = RUNS_DIR) -> None:
        """Creates a unique run directory under run_root and initialises the timings dict. Every Monitor instance isolates its JSON artifacts in a separate folder identified by a random ID."""
        self.run_id = uuid.uuid4().hex[:10]
        self.run_dir = Path(run_root) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.timings: dict[str, float] = {}

    @contextmanager
    def timeit(self, name: str):
        """Context manager — records elapsed time in ms under self.timings[name]."""
        start = time.perf_counter()
        yield
        self.timings[name] = round((time.perf_counter() - start) * 1000.0, 3)

    def write_json(self, filename: str, payload: dict[str, Any]) -> Path:
        """Writes payload as JSON into the run directory. Returns the file path."""
        path = self.run_dir / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


# -----------------------------------------------------------------------------
# VectorDatabase
# -----------------------------------------------------------------------------
class VectorDatabase:
    """PDF vector store: loads → chunks → embeds (HashingVectorizer) → FAISS IndexFlatIP."""

    def __init__(
        self,
        dim: int = EMBEDDING_DIM,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        monitor: Monitor | None = None,
    ) -> None:
        """Initialises the HashingVectorizer, text splitter, and an empty FAISS index. Call update_database() to load PDFs and build the index before querying."""
        self.dim = dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.monitor = monitor or Monitor()

        self.vectorizer = HashingVectorizer(n_features=self.dim, alternate_sign=False, lowercase=True)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.chunks: list[DocumentChunk] = []
        self.index: faiss.IndexFlatIP | None = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Transforms a list of strings into a contiguous float32 numpy array using HashingVectorizer. Used for both document indexing and query embedding."""
        x = self.vectorizer.transform(texts).astype(np.float32).toarray()
        return np.ascontiguousarray(x)

    def _load_pages(self, data_path: str) -> list[dict[str, Any]]:
        """Walks data_path/<topic>/*.pdf and returns a flat list of page records with text, source, topic, and page keys. Skips corrupted PDFs with a printed warning instead of crashing."""
        root = Path(data_path)
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Data path not found: {data_path}")

        pages: list[dict[str, Any]] = []
        for topic_dir in root.iterdir():
            if not topic_dir.is_dir():
                continue
            topic = topic_dir.name
            for pdf_path in topic_dir.glob("*.pdf"):
                try:
                    pages.extend(
                        {
                            "text": page.page_content,
                            "source": pdf_path.name,
                            "topic": topic,
                            "page": int(page.metadata.get("page", 0)),
                        }
                        for page in PyPDFLoader(str(pdf_path)).load()
                    )
                except Exception as e:
                    print(f"Skipping {pdf_path.name}: {e}")
        return pages

    def update_database(self, data_path: str) -> None:
        """Loads PDFs, chunks, embeds, and builds the FAISS index. Replaces any existing index."""
        with self.monitor.timeit("ingest_ms"):
            pages = self._load_pages(data_path)

        chunks: list[DocumentChunk] = []
        chunk_id = 0

        with self.monitor.timeit("chunk_ms"):
            for p in pages:
                for part in self.splitter.split_text(p["text"]):
                    if not part.strip() or _is_junk_chunk(part):
                        continue
                    chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        text=part,
                        source=p["source"],
                        topic=p["topic"],
                        page=p["page"],
                    ))
                    chunk_id += 1

        if not chunks:
            raise RuntimeError("No chunks produced from input data.")

        with self.monitor.timeit("embed_ms"):
            vectors = self._embed([c.text for c in chunks])

        with self.monitor.timeit("index_ms"):
            idx = faiss.IndexFlatIP(vectors.shape[1])
            idx.add(vectors)
            self.index = idx
            self.chunks = chunks

        self.monitor.write_json("00_config.json", {
            "embedding_model": EMBEDDING_MODEL_ID,
            "embedding_dim": self.dim,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k_default": TOP_K_DEFAULT,
        })
        pages_per_source = Counter(p["source"] for p in pages)
        chunks_per_source = Counter(c.source for c in chunks)
        self.monitor.write_json("10_ingest_summary.json", {
            "data_path": data_path,
            "n_pages": len(pages),
            "n_chunks": len(chunks),
            "ntotal": int(self.index.ntotal),
            "sources": [
                {"source": src, "pages": pages_per_source[src], "chunks": chunks_per_source[src]}
                for src in sorted(pages_per_source)
            ],
        })
        self.monitor.write_json("90_timings.json", self.monitor.timings)

    def recover(
        self,
        query: str,
        top_k: int = TOP_K_DEFAULT,
        topic: str | None = None,
    ) -> list[tuple[DocumentChunk, dict[str, Any], float]]:
        """Embeds the query and returns top-k chunks ranked by inner product score as (DocumentChunk, ref_dict, score) tuples. Optionally filters to a single topic folder by fetching extra candidates and post-filtering."""
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Index is empty. Call update_database(path) first.")

        with self.monitor.timeit("query_embed_ms"):
            qv = self._embed([query])

        fetch_k = (top_k * 5) if topic else top_k
        fetch_k = min(fetch_k, self.index.ntotal)

        with self.monitor.timeit("search_ms"):
            scores, indices = self.index.search(qv, fetch_k)

        out: list[tuple[DocumentChunk, dict[str, Any], float]] = []
        for s, i in zip(scores[0], indices[0]):
            c = self.chunks[int(i)]
            if topic is not None and c.topic != topic:
                continue
            ref = {"source": c.source, "topic": c.topic, "page": c.page, "chunk_id": c.chunk_id}
            out.append((c, ref, float(s)))
            if len(out) >= top_k:
                break

        debug = [
            {"rank": r, "score": round(score, 4), "source": c.source, "topic": c.topic, "chunk_id": c.chunk_id}
            for r, (c, _, score) in enumerate(out, 1)
        ]
        self.monitor.write_json("30_retrieval_debug.json", {
            "query": query,
            "top_k": top_k,
            "topic": topic,
            "results": debug,
            "timings": self.monitor.timings,
        })
        self.monitor.write_json("90_timings.json", self.monitor.timings)
        return out


# -----------------------------------------------------------------------------
# RAG
# -----------------------------------------------------------------------------
class RAG:
    """Full RAG pipeline: pre-guardrails → retrieval → LLM → post-guardrails → answer + citations."""

    def __init__(
        self,
        vector_db: VectorDatabase,
        llm: Any | None = None,
        monitor: Monitor | None = None,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        ambiguity_gap: float = AMBIGUITY_GAP,
    ) -> None:
        """Wires together a populated VectorDatabase and an LLM client into a single pipeline object. The llm can be None if only evaluate() will be called."""
        self.vector_db = vector_db
        self.llm = llm
        self.monitor = monitor or vector_db.monitor
        self.similarity_threshold = similarity_threshold
        self.ambiguity_gap = ambiguity_gap

    def _refuse(
        self,
        query: str,
        reason: str,
        stage: str,
        scores: list[float],
        model_name: str,
        sources_block: str,
        raw: str | None = None,
    ) -> str:
        """Logs the refusal to the generation debug file and returns a Blocked message string. Called whenever a pre- or post-generation guardrail fires."""
        self._write_generation_debug(query, "refused", stage, reason, scores, 0, model_name, raw)
        return f"Blocked ({stage}): {reason}\n\n{sources_block}"

    def _retrieve_pre_block(
        self, query: str, hits: list, scores: list[float]
    ) -> Tuple[str | None, str | None]:
        """Returns (stage, reason) to refuse, or (None, None) to proceed."""
        if not hits:
            return "pre", REFUSAL_MESSAGE
        guard_ok, guard_reason = _pre_guardrails_block(query)
        if not guard_ok:
            return "pre", guard_reason
        retrieval_ok, retrieval_reason = check_retrieval_quality(
            scores, self.similarity_threshold, self.ambiguity_gap
        )
        if not retrieval_ok:
            return "pre", retrieval_reason
        return None, None

    def _process_answer(
        self,
        raw_answer: str | None,
        query: str,
        scores: list[float],
        prompt_len: int,
        model_name: str,
        sources_block: str,
    ) -> str:
        """Applies post-generation guardrails to the raw LLM output and returns the final answer string. Handles insufficient context, data leakage, and uncertain language cases."""
        answer_text = (raw_answer or "")
        answer_snippet = answer_text[:2000]
        if raw_answer and "insufficient_context" in raw_answer.lower():
            self._write_generation_debug(
                query, "insufficient_context", "final", "INSUFFICIENT_CONTEXT",
                scores, prompt_len, model_name, answer_snippet,
            )
            return f"INSUFFICIENT_CONTEXT\n\n{sources_block}"
        answer_ok, answer_reason = check_generated_answer(answer_text)
        if not answer_ok and answer_reason:
            return self._refuse(
                query, answer_reason, "post", scores, model_name, sources_block, answer_snippet,
            )
        self._write_generation_debug(
            query, "answer", "final", None, scores, prompt_len, model_name, answer_snippet,
        )
        return answer_text.rstrip() + "\n\n" + sources_block

    def retrieve(self, query: str, top_k: int = TOP_K_DEFAULT) -> str:
        """
        Runs the full pipeline and returns a grounded answer with citations.
        Returns "Blocked (<stage>): <reason>" if any guardrail fires.
        """
        if self.llm is None:
            raise RuntimeError("LLM required for retrieve(). Pass an llm instance to RAG().")

        hits = self.vector_db.recover(query, top_k=top_k)
        scores = [s for _, _, s in hits]
        sources_block = "Sources:\n" + "\n".join(
            f"  [{i+1}] {_citation(ref)}" for i, (_, ref, _) in enumerate(hits)
        )
        model_name = getattr(self.llm, "model", type(self.llm).__name__)

        stage, reason = self._retrieve_pre_block(query, hits, scores)
        if stage is not None:
            return self._refuse(query, reason or REFUSAL_MESSAGE, stage, scores, model_name, sources_block)

        context = _format_context(hits)
        prompt = _grounded_prompt(context, query)
        try:
            raw_answer = self.llm.invoke(prompt)
        except Exception as e:
            return self._refuse(
                query, f"LLM error: {str(e)[:100]}", "post", scores, model_name, sources_block
            )
        return self._process_answer(raw_answer, query, scores, len(prompt), model_name, sources_block)

    def evaluate(self, eval_path: str, top_k: int = 5, use_topic: bool = True) -> dict[str, Any]:
        """Runs retrieval-only evaluation over a JSONL file and returns hit@1/3/5 and MRR@5 metrics. Does not call the LLM — each query is scored against gold_chunk_ids, gold_sources, or gold_keywords."""
        path = Path(eval_path)
        if not path.is_file():
            raise FileNotFoundError(f"Eval file not found or not a file: {eval_path}")

        n_eval = 0
        n_skipped = 0
        hit1_scores: list[float] = []
        hit3_scores: list[float] = []
        hit5_scores: list[float] = []
        mrr5_scores: list[float] = []
        failures: list[dict[str, Any]] = []

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue

            query = row.get("query") or row.get("question", "")
            if not query or not str(query).strip():
                n_skipped += 1
                continue

            query = str(query).strip()
            row_id = row.get("id")
            topic = row.get("topic") if use_topic else None
            gold_chunk_ids: list[int] = row.get("gold_chunk_ids") or []
            gold_sources: list[str] = row.get("gold_sources") or []
            gold_keywords: list[str] = row.get("gold_keywords") or []
            has_gold = bool(gold_chunk_ids or gold_sources or gold_keywords)

            hits = self.vector_db.recover(query, top_k=top_k, topic=topic)
            correct_ranks = [
                rank
                for rank, (chunk, ref, _) in enumerate(hits[:5], 1)
                if _eval_hit_correct(chunk, ref, gold_chunk_ids, gold_sources, gold_keywords)
            ]

            if has_gold:
                h1, h3, h5, mrr = _hit_mrr_scores(correct_ranks)
                hit1_scores.append(h1)
                hit3_scores.append(h3)
                hit5_scores.append(h5)
                mrr5_scores.append(mrr)
                if not correct_ranks:
                    failures.append({
                        "id": row_id,
                        "query": query,
                        "top_sources": [ref["source"] for _, ref, _ in hits],
                        "top_chunk_ids": [chunk.chunk_id for chunk, _, _ in hits],
                    })
            n_eval += 1

        report: dict[str, Any] = {
            "n": n_eval,
            "skipped": n_skipped,
            "hit@1_mean": _mean(hit1_scores),
            "hit@3_mean": _mean(hit3_scores),
            "hit@5_mean": _mean(hit5_scores),
            "mrr@5_mean": _mean(mrr5_scores),
            "failures": failures,
        }
        self.monitor.write_json("60_eval_report.json", report)
        return report

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
        """Writes generation outcome metadata to 40_generation_debug.json."""
        self.monitor.write_json("40_generation_debug.json", {
            "query": query,
            "outcome": outcome,
            "stage": stage,
            "reason": reason,
            "max_score": round(scores[0], 4) if scores else None,
            "score_gap": round(scores[0] - scores[1], 4) if len(scores) >= 2 else None,
            "prompt_length": prompt_length,
            "model_used": model_used,
            "raw_answer": (raw_answer or "")[:2000],
        })
