import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import faiss
from sklearn.feature_extraction.text import HashingVectorizer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Optional: real LLM (Week 3 style). If Ollama not installed, we will gracefully fall back.
try:
    from langchain_ollama import OllamaLLM
    _OLLAMA_AVAILABLE = True
except Exception:
    _OLLAMA_AVAILABLE = False


# =========================
# Data types (minimal)
# =========================

@dataclass
class Chunk:
    text: str
    source: str
    topic: str
    page: int


@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float
    rank: int

class Monitor:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex[:10]
        self.events: List[Dict[str, Any]] = []
        self.timings: Dict[str, float] = {}

    def timeit(self, name: str):
        class _Timer:
            def __init__(self, outer, n): self.outer, self.n, self.t0 = outer, n, None
            def __enter__(self): self.t0 = time.perf_counter()
            def __exit__(self, *args):
                self.outer.timings[self.n] = round((time.perf_counter() - self.t0) * 1000.0, 3)
        return _Timer(self, name)

    def log(self, event: str, payload: Dict[str, Any]):
        self.events.append({"event": event, "payload": payload})

    def write_json(self, filename: str, payload: Dict[str, Any]):
        path = self.artifacts_dir / f"{self.run_id}_{filename}"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return str(path)

def check_pii(query: str) -> bool:
    pii_keywords = [
        "email", "phone", "ssn", "social security",
        "credit card", "passport", "driver's license",
        "address", "zip code", "date of birth",
    ]
    q = query.lower()
    return any(k in q for k in pii_keywords)

def check_prompt_injection(query: str) -> bool:
    patterns = [
        "ignore previous instructions",
        "ignore all instructions",
        "system prompt",
        "developer mode",
        "jailbreak",
        "override",
        "do anything now",
        "you are now",
        "act as",
    ]
    q = query.lower()
    return any(p in q for p in patterns)

def check_toxicity(query: str) -> bool:
    # Very simple educational filter.
    keywords = ["hack", "steal passwords", "make a bomb", "explosive", "kill", "credit card fraud"]
    q = query.lower()
    return any(k in q for k in keywords)

def retrieval_quality_gate(scores: List[float], similarity_threshold: float, ambiguity_gap: float) -> Tuple[bool, Optional[str]]:
    if scores is None or len(scores) == 0:
        return False, "EMPTY_RETRIEVAL"
    top1 = float(scores[0])
    if top1 < similarity_threshold:
        return False, "NO_CONTEXT"
    if len(scores) >= 2:
        gap = float(scores[0] - scores[1])
        if gap < ambiguity_gap:
            return False, "AMBIGUOUS_RETRIEVAL"
    return True, None

class VectorDatabase:
    def __init__(
        self,
        embedding_dim: int = 768,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        max_pages_per_pdf: int = 20,
        artifacts_dir: str = "artifacts",
    ):
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_pages_per_pdf = max_pages_per_pdf

        self.vectorizer = HashingVectorizer(
            n_features=self.embedding_dim,
            alternate_sign=False,
            norm="l2",
        )

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.chunks: List[Chunk] = []
        self.monitor = Monitor(artifacts_dir=artifacts_dir)

    def _embed(self, texts: List[str]) -> np.ndarray:
        x = self.vectorizer.transform(texts).astype(np.float32).toarray()
        faiss.normalize_L2(x)
        return x

    def update_database(self, data_path: str) -> None:
        """
        data_path format expected:
          data/
            topicA/*.pdf
            topicB/*.pdf
        """
        base = Path(data_path)
        if not base.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")

        docs = []
        ingest_errors = []

        with self.monitor.timeit("ingest_ms"):
            for topic_dir in sorted(base.iterdir()):
                if not topic_dir.is_dir():
                    continue
                topic = topic_dir.name
                for pdf_path in sorted(topic_dir.glob("*.pdf")):
                    try:
                        pages = PyPDFLoader(str(pdf_path)).load()[: self.max_pages_per_pdf]
                        for p in pages:
                            docs.append({
                                "text": p.page_content,
                                "source": pdf_path.name,
                                "topic": topic,
                                "page": int(p.metadata.get("page", 0)),
                            })
                    except Exception as e:
                        ingest_errors.append({"source": pdf_path.name, "error": str(e)})

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        new_chunks: List[Chunk] = []
        with self.monitor.timeit("chunking_ms"):
            for d in docs:
                parts = splitter.split_text(d["text"])
                for part in parts:
                    part = (part or "").strip()
                    if part:
                        new_chunks.append(Chunk(
                            text=part,
                            source=d["source"],
                            topic=d["topic"],
                            page=d["page"],
                        ))

        with self.monitor.timeit("embed_ms"):
            vecs = self._embed([c.text for c in new_chunks])

        with self.monitor.timeit("index_add_ms"):
            self.index.reset()
            self.index.add(vecs)
            self.chunks = new_chunks

        # artifacts
        self.monitor.write_json("00_config.json", {
            "embedding_dim": self.embedding_dim,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_pages_per_pdf": self.max_pages_per_pdf,
        })
        self.monitor.write_json("10_ingest_summary.json", {
            "docs_count": len(docs),
            "chunks_count": len(new_chunks),
            "ingest_errors": ingest_errors[:20],
        })
        sample = [{
            "text_preview": c.text[:180],
            "source": c.source,
            "topic": c.topic,
            "page": c.page,
        } for c in new_chunks[:5]]
        self.monitor.write_json("20_chunks_sample.json", {"sample": sample})
        self.monitor.write_json("90_timings.json", self.monitor.timings)

    def recover(self, query: str, top_k: int = 5) -> List[RetrievalHit]:
        if len(self.chunks) == 0 or self.index.ntotal == 0:
            return []

        with self.monitor.timeit("query_embed_ms"):
            qv = self._embed([query])

        with self.monitor.timeit("faiss_search_ms"):
            scores, idxs = self.index.search(qv, top_k)

        scores_list = [float(s) for s in scores[0]]
        idx_list = [int(i) for i in idxs[0]]

        hits: List[RetrievalHit] = []
        for rank, (i, s) in enumerate(zip(idx_list, scores_list), start=1):
            if i < 0 or i >= len(self.chunks):
                continue
            hits.append(RetrievalHit(chunk=self.chunks[i], score=s, rank=rank))

        self.monitor.write_json("30_retrieval_debug.json", {
            "query": query,
            "top_k": top_k,
            "hits": [{
                "rank": h.rank,
                "score": h.score,
                "topic": h.chunk.topic,
                "source": h.chunk.source,
                "page": h.chunk.page,
                "text_preview": h.chunk.text[:180],
            } for h in hits],
            "timings": self.monitor.timings,
        })

        return hits


class RAG:
    def __init__(
        self,
        vector_db: VectorDatabase,
        model: str = "gemma3:4b",
        temperature: float = 0.2,
        similarity_threshold: float = 0.30,
        ambiguity_gap: float = 0.05,
    ):
        self.vector_db = vector_db
        self.monitor = vector_db.monitor  # 1-to-1 shared monitor
        self.similarity_threshold = similarity_threshold
        self.ambiguity_gap = ambiguity_gap

        self.model = model
        self.temperature = temperature
        self.llm = None
        if _OLLAMA_AVAILABLE:
            try:
                self.llm = OllamaLLM(model=self.model, temperature=self.temperature)
            except Exception:
                self.llm = None

    def _build_prompt(self, hits: List[RetrievalHit], query: str) -> str:
        ctx_lines = []
        for h in hits:
            ctx_lines.append(
                f"[Chunk {h.rank}] (score={h.score:.4f}, source={h.chunk.source}, page={h.chunk.page}, topic={h.chunk.topic})\n"
                f"{h.chunk.text}"
            )
        context = "\n\n".join(ctx_lines)

        prompt = f"""You are a strict grounded RAG assistant.
Rules:
1) Use ONLY the CONTEXT chunks.
2) If evidence is missing or weak, answer exactly: Insufficient evidence in retrieved context.
3) Add citations like [Chunk 1], [Chunk 2] for every factual claim.
4) Do not add external facts.

CONTEXT:
{context}

QUESTION: {query}
ANSWER:
"""
        self.monitor.write_json("40_prompt_debug.json", {
            "query": query,
            "prompt_preview": prompt[:1200],
            "n_hits": len(hits),
        })
        return prompt

    def retrieve(self, query: str, top_k: int = 5) -> str:
        # Pre-check guardrails
        if check_pii(query):
            self.monitor.write_json("50_guardrails_report.json", {"allowed": False, "reason": "PII_DETECTED"})
            return "Blocked by guardrails: PII_DETECTED"
        if check_prompt_injection(query):
            self.monitor.write_json("50_guardrails_report.json", {"allowed": False, "reason": "PROMPT_INJECTION"})
            return "Blocked by guardrails: PROMPT_INJECTION"
        if check_toxicity(query):
            self.monitor.write_json("50_guardrails_report.json", {"allowed": False, "reason": "TOXICITY"})
            return "Blocked by guardrails: TOXICITY"

        hits = self.vector_db.recover(query, top_k=top_k)
        scores = [h.score for h in hits]

        ok, reason = retrieval_quality_gate(scores, self.similarity_threshold, self.ambiguity_gap)
        if not ok:
            self.monitor.write_json("50_guardrails_report.json", {"allowed": False, "reason": reason, "scores": scores[:3]})
            return "Insufficient evidence in retrieved context."

        prompt = self._build_prompt(hits, query)

        # LLM call (must be blocked if guardrails fail)
        if self.llm is None:
            self.monitor.write_json("60_generation_output.json", {
                "llm_called": False,
                "reason": "LLM_NOT_AVAILABLE",
            })
            return "LLM not available locally. Retrieved context is ready; start Ollama to generate."

        with self.monitor.timeit("llm_ms"):
            answer = (self.llm.invoke(prompt) or "").strip()

        self.monitor.write_json("60_generation_output.json", {
            "llm_called": True,
            "model": self.model,
            "temperature": self.temperature,
            "answer_preview": answer[:400],
            "timings": self.monitor.timings,
        })

        return answer


# =========================
# Demo entrypoint (mentor-friendly)
# =========================

if __name__ == "__main__":
    # Adjust if your repo uses a different folder
    data_dir = "data"
    q = "What is RAG?"

    vd = VectorDatabase(embedding_dim=768, chunk_size=300, chunk_overlap=50, max_pages_per_pdf=20)
    vd.update_database(data_dir)

    rag = RAG(vd, model="gemma3:4b", temperature=0.2, similarity_threshold=0.30, ambiguity_gap=0.05)

    print("=" * 80)
    print("QUERY:", q)
    print("=" * 80)
    out = rag.retrieve(q, top_k=3)
    print(out)
    print("\nArtifacts run_id:", vd.monitor.run_id)
    print("Artifacts dir:", vd.monitor.artifacts_dir.resolve())