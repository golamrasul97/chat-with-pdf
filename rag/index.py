"""Phase 3 + 4 + 5 of the loop: embed -> store -> retrieve.

Chunks become vectors (embeddings), the vectors go into an in-memory FAISS
index, and at question time we pull back the handful of chunks whose vectors sit
closest to the question's vector. "Closest" is the entire retrieval bet: if the
right chunk isn't near the question in vector space, no LLM can save the answer.
"""

from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once and reuse it.

    The model weights take a second or two to load and are completely
    stateless, so we cache the instance rather than reloading per request or
    per session. (Downloaded once to ~/.cache on first ever run.)
    """
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def build_index(chunks: list[Document]) -> FAISS:
    """Embed every chunk and store the vectors in a fresh FAISS index.

    Returned index is held per-session in the UI, so one visitor's PDF is never
    searchable by another's.
    """
    return FAISS.from_documents(chunks, get_embeddings())


def retrieve(index: FAISS, question: str) -> list[tuple[Document, float]]:
    """Return the top-k chunks nearest the question, each with its distance.

    We keep the score (FAISS L2 distance — smaller means more similar) and
    surface it in the UI. That number is the first thing to look at when an
    answer is wrong: it indicates whether retrieval even found relevant text.
    """
    return index.similarity_search_with_score(question, k=settings.top_k)
