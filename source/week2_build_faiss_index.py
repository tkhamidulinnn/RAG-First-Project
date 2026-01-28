# Run: python week2_build_faiss_index.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer


# ============================================================
# Project paths
# ============================================================

def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(8):
        if (cur / "artifacts").exists():
            return cur
        if (cur / ".git").exists():
            return cur
        cur = cur.parent
    return start.resolve()


PROJECT_ROOT = find_project_root(Path(__file__).parent)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


# ============================================================
# RAG Knowledge Base
# ============================================================

DOCS = [
    {
        "id": "rag_01",
        "topic": "RAG",
        "text": """Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation. 
        Instead of relying solely on the knowledge stored in model parameters, 
        RAG systems retrieve relevant documents from an external knowledge base and use them as context for generating responses. 
        This approach reduces hallucinations by grounding the model's output in factual, retrieved information. 
        The typical RAG pipeline consists of three stages: indexing documents into a searchable format, 
        retrieving relevant passages given a query, and generating a response using the retrieved context. 
        RAG is particularly useful for knowledge-intensive tasks where the model needs access to up-to-date or domain-specific information that wasn't part of its training data."""
    },
    {
        "id": "rag_02",
        "topic": "RAG",
        "text": """The key advantage of RAG over fine-tuning is that the knowledge base can be updated without retraining the model. 
        This makes RAG cost-effective and flexible for enterprise applications. 
        RAG systems also provide transparency since retrieved sources can be cited, 
        allowing users to verify the information. Common challenges in RAG include retrieval quality, 
        context window limitations, and handling conflicting information from multiple sources. 
        Advanced RAG architectures may include query rewriting, multi-hop retrieval, and fusion techniques to improve answer quality."""
    },
    {
        "id": "chunking_01",
        "topic": "Chunking",
        "text": """Chunking is the process of splitting documents into smaller pieces for indexing and retrieval. 
        The chunk size significantly impacts retrieval quality. 
        Small chunks (100-200 tokens) provide precise matches but may lose context. 
        Large chunks (500-1000 tokens) preserve context but may include irrelevant information. 
        Overlap between chunks helps maintain continuity across chunk boundaries. 
        Common chunking strategies include fixed-size chunking, sentence-based splitting, and semantic chunking based on topic boundaries. 
        The optimal chunk size depends on the embedding model's context window, the nature of queries, and the document structure."""
    },
    {
        "id": "chunking_02",
        "topic": "Chunking",
        "text": """Recursive character text splitting is a popular chunking method that tries to split on natural boundaries like paragraphs and sentences before falling back to character-level splits. 
        This preserves semantic coherence within chunks. 
        Chunk overlap (typically 10-20% of chunk size) ensures that information spanning chunk boundaries is not lost. 
        For structured documents like code or markdown, specialized splitters that respect syntax boundaries produce better results. 
        Parent-child chunking stores both small chunks for precise retrieval and their parent documents for expanded context."""
    },
    {
        "id": "embeddings_01",
        "topic": "Embeddings",
        "text": """Text embeddings are dense vector representations that capture semantic meaning. 
        In RAG systems, both documents and queries are converted to embeddings, 
        and similarity search finds the most relevant documents. 
        Popular embedding models include OpenAI's text-embedding-ada-002, 
        Sentence Transformers (like all-MiniLM-L6-v2), and BGE models. 
        The embedding dimension affects storage requirements and search speed. 
        Normalized embeddings allow using inner product as a similarity metric, 
        which is computationally efficient. 
        The choice of embedding model significantly impacts retrieval quality and should match the domain of your documents."""
    },
    {
        "id": "vectordb_01",
        "topic": "VectorDB",
        "text": """Vector databases store embeddings and enable fast similarity search at scale. 
        FAISS (Facebook AI Similarity Search) is a popular open-source library for efficient similarity search. 
        It supports various index types: IndexFlatIP for exact search, IndexIVF for approximate search with clustering, 
        and IndexHNSW for graph-based approximate search. For production systems, managed vector databases like Pinecone, 
        Weaviate, Milvus, and Qdrant provide additional features like filtering, hybrid search, and automatic scaling. 
        The choice between exact and approximate nearest neighbor search depends on the dataset size and latency requirements."""
    },
    {
        "id": "hyde_01",
        "topic": "HyDE",
        "text": """HyDE (Hypothetical Document Embeddings) is an advanced retrieval technique that improves search quality 
        by generating a hypothetical answer before retrieval. 
        Instead of embedding the query directly, HyDE uses an LLM to generate a hypothetical document that would answer the query,
          then embeds this generated document for retrieval. 
          This bridges the semantic gap between short queries and longer document passages. 
          HyDE is particularly effective when queries are vague or use different terminology than the indexed documents. 
          The technique adds latency due to the generation step but often significantly improves retrieval accuracy."""
    },
    {
        "id": "evaluation_01",
        "topic": "Evaluation",
        "text": """Evaluating RAG systems requires measuring both retrieval quality and generation quality. 
        Retrieval metrics include Hit@K (whether relevant documents appear in top K results), 
        Mean Reciprocal Rank (MRR), and Normalized Discounted Cumulative Gain (NDCG). Generation quality can be measured using BLEU, 
        ROUGE, or model-based metrics like faithfulness and relevance scores. End-to-end evaluation often uses human judgment or LLM-as-judge approaches. 
        Important aspects to evaluate include factual accuracy, completeness, relevance to the query, and proper attribution of sources. 
        A/B testing with real users provides the most reliable quality signal."""
    },
]


# ============================================================
# Configuration
# ============================================================

CHUNK_CONFIGS = [
    {"chunk_size": 200, "overlap": 20},
    {"chunk_size": 500, "overlap": 50},
    {"chunk_size": 800, "overlap": 80},
]

MODELS = {
    "MiniLM": "all-MiniLM-L6-v2",
    "MPNet": "all-mpnet-base-v2",
}

EVAL_QUERIES = [
    # Direct queries (easy)
    {"query": "Explain retrieval augmented generation in simple terms", "expected_topic": "RAG"},
    {"query": "How does retrieval reduce hallucinations?", "expected_topic": "RAG"},
    {"query": "What is the advantage of RAG over fine-tuning?", "expected_topic": "RAG"},
    {"query": "What is chunking and why do we use overlap?", "expected_topic": "Chunking"},
    {"query": "How does chunk size affect retrieval quality?", "expected_topic": "Chunking"},
    {"query": "What is recursive character text splitting?", "expected_topic": "Chunking"},
    {"query": "What are text embeddings and how do they work?", "expected_topic": "Embeddings"},
    {"query": "Which embedding models are popular for RAG?", "expected_topic": "Embeddings"},
    {"query": "What is FAISS and how does it work?", "expected_topic": "VectorDB"},
    {"query": "What are the different FAISS index types?", "expected_topic": "VectorDB"},
    {"query": "What is HyDE in retrieval?", "expected_topic": "HyDE"},
    {"query": "How does hypothetical document embedding improve search?", "expected_topic": "HyDE"},
    {"query": "How do we evaluate retrieval quality?", "expected_topic": "Evaluation"},
    {"query": "What is Mean Reciprocal Rank (MRR)?", "expected_topic": "Evaluation"},
    {"query": "Why does adding retrieved context help LLM answers stay accurate?", "expected_topic": "RAG"},
    {"query": "Does overlap always improve retrieval precision?", "expected_topic": "Chunking"},

    # =========================================================================
    # TRAP QUERIES (harder, paraphrased, conceptual traps)
    # Designed to lower hit@1 and MRR for realistic evaluation
    # =========================================================================

    # RAG vs Fine-tuning (conceptual traps)
    {"query": "Should I retrain my model or just add a search layer?", "expected_topic": "RAG"},
    {"query": "When is updating model weights better than external retrieval?", "expected_topic": "RAG"},
    {"query": "Can retrieval completely replace parameter updates for domain adaptation?", "expected_topic": "RAG"},

    # Chunking trade-offs and edge cases
    {"query": "What happens if my chunks are too small to contain a complete answer?", "expected_topic": "Chunking"},
    {"query": "Does splitting documents always improve search precision?", "expected_topic": "Chunking"},
    {"query": "How do I handle tables and code blocks when segmenting documents?", "expected_topic": "Chunking"},

    # HyDE limitations and misconceptions
    {"query": "Is generating a fake answer before search the same as fine-tuning?", "expected_topic": "HyDE"},
    {"query": "Why would I want the LLM to hallucinate before retrieval?", "expected_topic": "HyDE"},
    {"query": "Does HyDE work when the query uses jargon the LLM doesn't know?", "expected_topic": "HyDE"},

    # Embeddings and similarity metric nuances
    {"query": "Why do two semantically similar sentences sometimes have low cosine scores?", "expected_topic": "Embeddings"},
    {"query": "Can inner product and cosine similarity give different ranking results?", "expected_topic": "Embeddings"},
]


# ============================================================
# Chunking
# ============================================================

def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return [c for c in chunks if c]


def make_chunks(docs, cfg):
    chunks = []
    for d in docs:
        parts = chunk_text(d["text"], cfg["chunk_size"], cfg["overlap"])
        for i, p in enumerate(parts):
            chunks.append({
                "doc_id": d["id"],
                "topic": d["topic"],
                "chunk_id": f"{d['id']}::c{i}",
                "text": p
            })
    return chunks


# ============================================================
# FAISS Index (cosine similarity via normalized embeddings)
# ============================================================

def build_index(texts, model_name):
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, normalize_embeddings=True)
    emb = np.array(emb, dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return model, index


def retrieve(model, index, query, top_k):
    q = model.encode([query], normalize_embeddings=True)
    q = np.array(q, dtype="float32")
    scores, idxs = index.search(q, top_k)
    return scores[0], idxs[0]


# ============================================================
# Metrics
# ============================================================

def hit_at_k(expected, retrieved_topics, k):
    return 1.0 if expected in retrieved_topics[:k] else 0.0


def mrr(expected, retrieved_topics):
    for i, t in enumerate(retrieved_topics, start=1):
        if t == expected:
            return 1.0 / i
    return 0.0


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Week 2: FAISS Retrieval Pipeline")
    print("=" * 60)
    print(f"Documents: {len(DOCS)}")
    print(f"Topics: {set(d['topic'] for d in DOCS)}")
    print(f"Queries: {len(EVAL_QUERIES)}")
    print(f"Chunk configs: {len(CHUNK_CONFIGS)}")
    print(f"Models: {list(MODELS.keys())}")
    print()

    rows = []
    debug = []

    for cfg in CHUNK_CONFIGS:
        chunks = make_chunks(DOCS, cfg)
        texts = [c["text"] for c in chunks]
        topics = [c["topic"] for c in chunks]

        print(f"Chunk config: size={cfg['chunk_size']}, overlap={cfg['overlap']} -> {len(chunks)} chunks")

        for model_label, model_name in MODELS.items():
            print(f"  Loading model: {model_label}...")
            model, index = build_index(texts, model_name)

            for q in EVAL_QUERIES:
                scores, idxs = retrieve(model, index, q["query"], top_k=5)
                retrieved_topics = [topics[i] for i in idxs]
                gap_1_2 = float(scores[0] - scores[1]) if len(scores) > 1 else None

                rows.append({
                    "chunk_size": cfg["chunk_size"],
                    "overlap": cfg["overlap"],
                    "model": model_label,
                    "query": q["query"],
                    "expected": q["expected_topic"],
                    "hit@1": hit_at_k(q["expected_topic"], retrieved_topics, 1),
                    "hit@3": hit_at_k(q["expected_topic"], retrieved_topics, 3),
                    "hit@5": hit_at_k(q["expected_topic"], retrieved_topics, 5),
                    "mrr": mrr(q["expected_topic"], retrieved_topics),
                    "top_topics": retrieved_topics,
                    "top1_score": float(scores[0]),
                    "gap_1_2": gap_1_2,
                })

                for rank in range(3):
                    debug.append({
                        "chunk_size": cfg["chunk_size"],
                        "overlap": cfg["overlap"],
                        "model": model_label,
                        "query": q["query"],
                        "rank": rank + 1,
                        "score": float(scores[rank]),
                        "topic": topics[idxs[rank]],
                        "chunk_id": chunks[idxs[rank]]["chunk_id"],
                        "text_preview": chunks[idxs[rank]]["text"][:200].replace("\n", " "),
                    })

    df = pd.DataFrame(rows)
    df_debug = pd.DataFrame(debug)

    # Summary
    summary = (
        df.groupby(["chunk_size", "overlap", "model"])
          .agg({"hit@1": "mean", "hit@3": "mean", "hit@5": "mean", "mrr": "mean", "top1_score": "mean", "gap_1_2": "mean"})
          .reset_index()
          .sort_values(["mrr", "hit@1"], ascending=False)
    )

    # Save artifacts
    results_path = ARTIFACTS_DIR / "week2_results.csv"
    summary_path = ARTIFACTS_DIR / "week2_summary.csv"
    debug_path = ARTIFACTS_DIR / "week2_debug.csv"

    df.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    df_debug.to_csv(debug_path, index=False)

    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(summary.round(3).to_string(index=False))
    print()
    print("Saved:", results_path)
    print("Saved:", summary_path)
    print("Saved:", debug_path)
    print()
    print("=" * 60)
    print("IMPORTANT NOTE")
    print("=" * 60)
    print("MRR=1 is caused by an easy evaluation setup (homogeneous corpus + direct queries),")
    print("not by a perfect retriever.")
    print()


if __name__ == "__main__":
    main()
