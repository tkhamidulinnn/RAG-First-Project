# Week 1: PDF Embedding Benchmark
# Evaluate embedding models on multi-topic PDF retrieval.
# Focus: embeddings, similarity metrics, retrieval correctness.
# NO LLM generation, NO reranking, NO advanced RAG.

import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =============================================================================
# Configuration
# =============================================================================

def get_project_root() -> Path:
    """
    Determine project root for both .py scripts and notebooks.
    """
    try:
        # Works in .py
        return Path(__file__).resolve().parents[1]
    except NameError:
        # Works in notebooks
        return Path.cwd().parent

PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Data dir exists: {DATA_DIR.exists()}")
print(f"Artifacts dir exists: {ARTIFACTS_DIR.exists()}")

# Chunking parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval parameter
K = 3

# Models to benchmark
MODELS = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
    "E5-base": "intfloat/e5-base-v2",
    "BGE-base": "BAAI/bge-base-en-v1.5",
    "GTE-base": "thenlper/gte-base",
}

# Test queries: (query_text, expected_topic)
# Ground truth = folder name containing relevant documents
QUERIES = [
    # RAG topic
    ("What is retrieval augmented generation?", "RAG"),
    ("How does HyDE improve retrieval?", "RAG"),
    ("What are LangChain retrieval methods?", "RAG"),
    # GIT topic
    ("How to create a git branch?", "GIT"),
    ("What is git rebase?", "GIT"),
    ("How to resolve merge conflicts?", "GIT"),
    # GCP topic
    ("What is Google Cloud Platform?", "GCP"),
    ("How to use gcloud CLI?", "GCP"),
    ("What are GCP security features?", "GCP"),
]

# =============================================================================
# 1) PDF Loading
# =============================================================================

def load_pdfs(data_dir: Path) -> list:
    """
    Load all PDFs from topic subfolders.
    Adds metadata: topic (folder name), source (filename), page.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    docs = []
    topics_found = []

    for topic_dir in sorted(data_dir.iterdir()):
        if not topic_dir.is_dir():
            continue

        topic = topic_dir.name
        pdf_files = list(topic_dir.glob("*.pdf"))

        if not pdf_files:
            print(f"  WARNING: No PDFs in {topic}/")
            continue

        topics_found.append(topic)
        print(f"\n[{topic}]")

        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_path))
                pdf_docs = loader.load()
                for doc in pdf_docs:
                    doc.metadata["topic"] = topic
                    doc.metadata["source"] = pdf_path.name
                    docs.append(doc)
                print(f"  + {pdf_path.name} ({len(pdf_docs)} pages)")
            except Exception as e:
                print(f"  ERROR: {pdf_path.name} - {e}")

    print(f"\nLoaded: {len(docs)} pages from {len(topics_found)} topics: {topics_found}")

    if not docs:
        raise ValueError("No documents loaded. Check your data directory.")

    return docs

# =============================================================================
# 2) Chunking
# =============================================================================

def chunk_docs(docs: list, chunk_size: int, overlap: int) -> list:
    """Split documents into chunks, preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    # Count chunks per topic
    topic_counts = {}
    for c in chunks:
        topic = c.metadata["topic"]
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    print(f"Created {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    print(f"Per topic: {topic_counts}")

    if not chunks:
        raise ValueError("No chunks created. Check chunk_size parameter.")

    return chunks

# =============================================================================
# 3) Similarity Functions
# =============================================================================

def normalize_vectors(v: np.ndarray) -> np.ndarray:
    """L2 normalize vectors. Handles both 1D and 2D arrays."""
    if v.ndim == 1:
        return v / (np.linalg.norm(v) + 1e-9)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

def cosine_similarity(doc_emb: np.ndarray, query_emb: np.ndarray) -> np.ndarray:
    """Cosine similarity = dot product of normalized vectors."""
    doc_norm = normalize_vectors(doc_emb)
    query_norm = normalize_vectors(query_emb)
    return doc_norm @ query_norm

def dot_product(doc_emb: np.ndarray, query_emb: np.ndarray) -> np.ndarray:
    """Raw dot product (no normalization)."""
    return doc_emb @ query_emb

def euclidean_distance(doc_emb: np.ndarray, query_emb: np.ndarray) -> np.ndarray:
    """Negative Euclidean distance (higher = more similar)."""
    return -np.linalg.norm(doc_emb - query_emb, axis=1)

SIMILARITY_FUNCTIONS = {
    "cosine": cosine_similarity,
    "dot_product": dot_product,
    "euclidean": euclidean_distance,
}

# =============================================================================
# 4) Evaluation Metrics
# =============================================================================

def hit_at_k(retrieved_topics: list, expected_topic: str, k: int) -> float:
    """Returns 1.0 if expected topic appears in top-k, else 0.0."""
    return 1.0 if expected_topic in retrieved_topics[:k] else 0.0

def mrr(retrieved_topics: list, expected_topic: str) -> float:
    """Mean Reciprocal Rank: 1/position of first correct result."""
    for i, topic in enumerate(retrieved_topics, start=1):
        if topic == expected_topic:
            return 1.0 / i
    return 0.0

def topic_accuracy_at_k(retrieved_topics: list, expected_topic: str, k: int) -> float:
    """Fraction of top-k results matching expected topic."""
    top_k = retrieved_topics[:k]
    if not top_k:
        return 0.0
    return sum(1 for t in top_k if t == expected_topic) / len(top_k)

# =============================================================================
# 5) Model-Specific Formatting
# =============================================================================

def format_query(text: str, model_id: str) -> str:
    """Apply model-specific query prefix."""
    model_lower = model_id.lower()
    if "e5" in model_lower:
        return f"query: {text}"
    if "bge" in model_lower:
        return f"Represent this sentence for searching relevant passages: {text}"
    return text

def format_passage(text: str, model_id: str) -> str:
    """Apply model-specific passage prefix."""
    if "e5" in model_id.lower():
        return f"passage: {text}"
    return text

# =============================================================================
# 6) Main Benchmark
# =============================================================================

def run_benchmark(chunks: list, queries: list, k: int = 3, verbose: bool = True) -> pd.DataFrame:
    """
    Run retrieval benchmark across all models and similarity functions.
    Returns DataFrame with per-query results.
    """
    results = []
    # Track first query of each topic for debug output
    debug_queries = {q[1]: q[0] for q in queries}  # topic -> first query

    for model_name, model_id in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        model = SentenceTransformer(model_id)

        # Encode all chunks ONCE (not twice!)
        print("Encoding chunks...")
        chunk_texts = [format_passage(c.page_content, model_id) for c in chunks]
        doc_embeddings = model.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=False)

        for query_text, expected_topic in queries:
            # Encode query
            formatted_query = format_query(query_text, model_id)
            query_embedding = model.encode(formatted_query, convert_to_numpy=True)

            for sim_name, sim_fn in SIMILARITY_FUNCTIONS.items():
                # Compute similarity scores
                scores = sim_fn(doc_embeddings, query_embedding)

                # Get top-k indices
                top_indices = np.argsort(scores)[::-1][:k]
                top_topics = [chunks[i].metadata["topic"] for i in top_indices]
                top_scores = scores[top_indices]

                # Compute metrics
                hit = hit_at_k(top_topics, expected_topic, k)
                rr = mrr(top_topics, expected_topic)
                acc = topic_accuracy_at_k(top_topics, expected_topic, k)

                results.append({
                    "model": model_name,
                    "similarity": sim_name,
                    "query": query_text[:40],
                    "expected": expected_topic,
                    f"hit@{k}": hit,
                    "mrr": rr,
                    f"topic_acc@{k}": acc,
                })

                # Debug: show details for first query of each topic (cosine only)
                is_first_query_of_topic = debug_queries.get(expected_topic) == query_text
                if verbose and sim_name == "cosine" and is_first_query_of_topic:
                    print(f"\nQuery: \"{query_text}\"")
                    print(f"Expected topic: {expected_topic}")
                    print(f"Top-{k} results:")
                    for rank, idx in enumerate(top_indices, 1):
                        chunk = chunks[idx]
                        snippet = chunk.page_content[:60].replace('\n', ' ')
                        src = chunk.metadata['source'][:30]
                        print(f"  {rank}. [{chunk.metadata['topic']}] {src}")
                        print(f"     score={top_scores[rank-1]:.4f} | \"{snippet}...\"")
                    print(f"Metrics: hit@{k}={hit:.0f}, MRR={rr:.2f}, topic_acc={acc:.0%}")

    return pd.DataFrame(results)

# =============================================================================
# 7) Main Execution
# =============================================================================

def main():
    """Main entry point."""
    # Setup
    print("="*60)
    print("Week 1: PDF Embedding Benchmark")
    print("="*60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data dir: {DATA_DIR} (exists: {DATA_DIR.exists()})")

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # 1. Load PDFs
    print("\n" + "="*60)
    print("STEP 1: Loading PDFs")
    print("="*60)
    docs = load_pdfs(DATA_DIR)

    # 2. Chunk documents
    print("\n" + "="*60)
    print("STEP 2: Chunking Documents")
    print("="*60)
    chunks = chunk_docs(docs, CHUNK_SIZE, CHUNK_OVERLAP)

    # 3. Run benchmark
    print("\n" + "="*60)
    print("STEP 3: Running Benchmark")
    print("="*60)
    df = run_benchmark(chunks, QUERIES, K)

    # 4. Print summaries
    print("\n" + "="*60)
    print("RESULTS: By Model + Similarity")
    print("="*60)
    summary = df.groupby(["model", "similarity"])[[f"hit@{K}", "mrr", f"topic_acc@{K}"]].mean()
    print(summary.round(3).to_string())

    print("\n" + "="*60)
    print("RESULTS: By Model (averaged over all similarities)")
    print("="*60)
    model_summary = df.groupby("model")[[f"hit@{K}", "mrr", f"topic_acc@{K}"]].mean()
    print(model_summary.sort_values("mrr", ascending=False).round(3).to_string())

    # 5. Save artifacts
    df.to_csv(ARTIFACTS_DIR / "week1_results.csv", index=False)
    summary.to_csv(ARTIFACTS_DIR / "week1_summary.csv")
    print(f"\nArtifacts saved to: {ARTIFACTS_DIR}")

    return df, summary

if __name__ == "__main__":
    main()
