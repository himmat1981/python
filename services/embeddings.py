from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL

# Singleton — loaded once at startup, reused for every request
_embedder = None

def get_embedder() -> SentenceTransformer:
    """Return the embedding model, loading it if not already loaded."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def encode(text: str) -> list:
    """Convert text to a vector embedding."""
    return get_embedder().encode(text).tolist()