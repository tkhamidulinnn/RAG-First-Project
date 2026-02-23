from __future__ import annotations

OLLAMA_MODEL = "gemma3:4b"
OLLAMA_TEMPERATURE = 0.2


class OllamaGemmaClient:
    """
    Thin wrapper around LangChain's OllamaLLM configured for gemma3:4b.

    Responsibilities:
    - Instantiate the Ollama LLM with a fixed model and temperature.
    - Expose a single invoke() method for prompt-to-text generation.

    This class is kept separate from the RAG framework so that the LLM
    dependency can be swapped without touching tragframe.py.
    """

    def __init__(self, model: str = OLLAMA_MODEL, temperature: float = OLLAMA_TEMPERATURE) -> None:
        """
        Instantiate the Ollama LLM client.

        Parameters:
            model (str): Ollama model name to use.
            temperature (float): Sampling temperature for generation.

        Returns:
            None

        Raises:
            RuntimeError: If langchain-ollama is not installed or Ollama is not running.
        """
        self.model = model
        try:
            from langchain_ollama import OllamaLLM
            self._llm = OllamaLLM(model=self.model, temperature=temperature)
        except Exception as e:
            raise RuntimeError(
                "Ollama is required. Run: pip install langchain-ollama "
                "&& ollama pull gemma3:4b && ollama serve"
            ) from e

    def invoke(self, prompt: str) -> str:
        """
        Send a prompt to the LLM and return the stripped text response.

        Parameters:
            prompt (str): The full prompt string.

        Returns:
            str: The model's text response.
        """
        return (self._llm.invoke(prompt) or "").strip()
