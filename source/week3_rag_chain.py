"""
Week 3: RAG Chain
Combines retrieval context with LLM generation using prompt templates.
"""

import time
import hashlib
from typing import Optional

from week3_llm_client import build_llm, DEFAULT_MODEL, DEFAULT_TEMPERATURE
from week3_prompt_templates import get_template, TEMPLATES


def format_context(
    chunks: list[str],
    scores: Optional[list[float]] = None,
    max_chars: int = 7000,
) -> str:
    """
    Format retrieved chunks into a context string for the prompt.

    Args:
        chunks: List of text chunks
        scores: Optional similarity scores for each chunk
        max_chars: Maximum characters to include

    Returns:
        Formatted context string
    """
    parts = []
    for i, ch in enumerate(chunks):
        score_str = f" (score={scores[i]:.4f})" if scores is not None else ""
        parts.append(f"[Chunk {i+1}{score_str}]\n{ch}")
    ctx = "\n\n".join(parts)
    return ctx[:max_chars]


def stable_id(*parts) -> str:
    """Generate a stable hash ID from input parts."""
    s = "||".join(map(str, parts))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def run_rag_chain(
    question: str,
    chunks: list[str],
    template_name: str = "qa_strict",
    scores: Optional[list[float]] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    topic: str = "RAG notes",
) -> dict:
    """
    Run the full RAG chain: format context + apply prompt + generate.

    Args:
        question: User question
        chunks: Retrieved text chunks
        template_name: One of 'qa_strict', 'summary', 'reasoning'
        scores: Optional similarity scores
        model: LLM model name
        temperature: Sampling temperature
        topic: Topic for summary template

    Returns:
        dict with generation results and metadata
    """
    llm = build_llm(model=model, temperature=temperature)
    prompt = get_template(template_name)

    context = format_context(chunks, scores)

    # Build prompt kwargs based on template variables
    kwargs = {"context": context}
    if "question" in prompt.input_variables:
        kwargs["question"] = question
    if "topic" in prompt.input_variables:
        kwargs["topic"] = topic

    prompt_text = prompt.format(**kwargs)

    # Generate
    t0 = time.time()
    answer = llm.invoke(prompt_text)
    latency = round(time.time() - t0, 3)

    answer = (answer or "").strip()

    return {
        "run_id": stable_id(question, template_name, model, temperature),
        "question": question,
        "template": template_name,
        "model": model,
        "temperature": temperature,
        "latency_s": latency,
        "prompt_chars": len(prompt_text),
        "context_chars": len(context),
        "n_chunks": len(chunks),
        "answer": answer,
        "idk_flag": "i don't know based on the provided context" in answer.lower(),
        "has_citation_flag": "[chunk" in answer.lower(),
    }


if __name__ == "__main__":
    # Demo with sample chunks
    sample_chunks = [
        "RAG combines retrieval with generation to reduce hallucinations.",
        "FAISS is a library for efficient similarity search.",
    ]

    print("Testing RAG chain with sample chunks...")
    print("=" * 50)

    for template in TEMPLATES.keys():
        result = run_rag_chain(
            question="What is RAG?",
            chunks=sample_chunks,
            template_name=template,
        )
        print(f"\nTemplate: {template}")
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Latency: {result['latency_s']}s")
