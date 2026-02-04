"""
Week 3: Prompt Templates for RAG
Three task-specific templates: Q&A, Summary, Reasoning.
"""

from langchain_core.prompts import PromptTemplate


# =============================================================================
# 1. Strict Q&A Prompt
# =============================================================================
# - Answer using ONLY context
# - If missing: exact refusal message
# - Require chunk citations

QA_STRICT_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a RAG assistant.\n"
        "You MUST answer using ONLY the provided CONTEXT.\n"
        "If the answer is not in the context, reply exactly:\n"
        "\"I don't know based on the provided context.\"\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION:\n{question}\n\n"
        "RULES:\n"
        "- No outside knowledge.\n"
        "- Be concise.\n"
        "- Cite chunks like [Chunk 2].\n\n"
        "ANSWER:\n"
    )
)


# =============================================================================
# 2. Structured Summary Prompt
# =============================================================================
# - Extract key points, definitions, practical notes, and gaps

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["context", "topic"],
    template=(
        "Summarize the provided context about: {topic}\n\n"
        "CONTEXT:\n{context}\n\n"
        "OUTPUT FORMAT:\n"
        "- Key points (3–7 bullets)\n"
        "- Definitions (if any)\n"
        "- Practical notes (if any)\n"
        "- Missing information / open questions (if any)\n\n"
        "SUMMARY:\n"
    )
)


# =============================================================================
# 3. Grounded Reasoning Prompt
# =============================================================================
# - Provide short reasoning steps tied to evidence
# - Avoid outside knowledge

REASONING_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an analyst. Use ONLY the provided context.\n"
        "Do not add outside knowledge.\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION:\n{question}\n\n"
        "OUTPUT FORMAT:\n"
        "1) Answer (1–3 sentences)\n"
        "2) Evidence: cite chunks like [Chunk 1]\n"
        "3) Reasoning: 3–6 short bullet steps tied to evidence\n\n"
        "RESPONSE:\n"
    )
)


# =============================================================================
# Template Registry
# =============================================================================

TEMPLATES = {
    "qa_strict": QA_STRICT_PROMPT,
    "summary": SUMMARY_PROMPT,
    "reasoning": REASONING_PROMPT,
}


def get_template(name: str) -> PromptTemplate:
    """
    Get a prompt template by name.

    Args:
        name: One of 'qa_strict', 'summary', 'reasoning'

    Returns:
        PromptTemplate instance

    Raises:
        KeyError: If template name not found
    """
    if name not in TEMPLATES:
        raise KeyError(f"Unknown template: {name}. Available: {list(TEMPLATES.keys())}")
    return TEMPLATES[name]


def list_templates() -> list[str]:
    """Return list of available template names."""
    return list(TEMPLATES.keys())
