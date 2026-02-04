"""
Week 3: LLM Client for Ollama
Provides a simple interface to call local LLM via Ollama.
"""

import time
from langchain_ollama import OllamaLLM


# Default configuration
DEFAULT_MODEL = "gemma3:4b"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.2


def build_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    base_url: str = DEFAULT_BASE_URL,
) -> OllamaLLM:
    """
    Build a LangChain-compatible Ollama LLM.

    Args:
        model: Model name as shown in `ollama list`
        temperature: Sampling temperature (0.0 = deterministic)
        base_url: Ollama server URL

    Returns:
        OllamaLLM instance
    """
    return OllamaLLM(
        model=model,
        temperature=temperature,
        base_url=base_url,
        validate_model_on_init=True,
    )


def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:
    """
    Calls Ollama and returns generation result with metadata.

    Args:
        prompt: The prompt text to send to the LLM
        model: Model name
        temperature: Sampling temperature

    Returns:
        dict with keys: text, latency, model
    """
    llm = build_llm(model=model, temperature=temperature)

    t0 = time.time()
    text = llm.invoke(prompt)
    latency = round(time.time() - t0, 3)

    return {
        "text": (text or "").strip(),
        "latency": latency,
        "model": model,
    }


if __name__ == "__main__":
    # Smoke test
    result = generate("Reply with one word: OK")
    print(f"Response: {result['text']}")
    print(f"Latency: {result['latency']}s")
    print(f"Model: {result['model']}")
