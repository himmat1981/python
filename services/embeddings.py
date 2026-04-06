"""
services/embeddings.py

LangChain HuggingFaceEmbeddings for query encoding.
Consistent with vectorstore.py — same model, same LangChain wrapper.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBED_MODEL

# Singleton — loaded once at startup, reused for every request
_embedder = None


def get_embedder() -> HuggingFaceEmbeddings:
    """Return the LangChain HuggingFaceEmbeddings model, loading it if needed."""
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(
            model_name    = EMBED_MODEL,
            model_kwargs  = {"device": "cpu"},
            encode_kwargs = {"normalize_embeddings": True},
        )
    return _embedder


def encode(text: str) -> list:
    """Convert text to a vector embedding using LangChain HuggingFaceEmbeddings."""
    return get_embedder().embed_query(text)
