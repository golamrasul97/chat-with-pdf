"""All configuration in one place, read from the environment.

The whole point of this file: the *only* thing that changes between running
locally (Ollama) and running on the Hugging Face Space (Groq) is environment
variables. No code branches on "am I local or hosted?" — we just point the same
OpenAI-compatible client at a different base_url/model/key.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load a local .env if present. On the Space there is no .env — the platform
# injects Variables and Secrets directly — so this is simply a no-op there.
load_dotenv()


def _resolve_api_key() -> str:
    """Pick the API key, tolerating both local and Space naming.

    We check LLM_API_KEY first (explicit), then GROQ_API_KEY (the name the
    Hugging Face Space Secret uses), then fall back to a dummy string that
    Ollama's OpenAI-compatible endpoint accepts but ignores. This is why local
    dev needs no real key.
    """
    return (
        os.getenv("LLM_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or "ollama"
    )


@dataclass(frozen=True)
class Settings:
    # --- LLM (generation) ---
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.2")
    llm_api_key: str = _resolve_api_key()

    # --- Embeddings (retrieval) ---
    # Small, fast, CPU-only, no API key. The retrieval quality ceiling of this
    # whole app is set here — swap this to trade speed for accuracy.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Chunking + retrieval knobs ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "4"))


# One shared instance imported everywhere else.
settings = Settings()
