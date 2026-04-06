"""
services/vectorstore.py

Step 1: LangChain handles node/store only.
  - RecursiveCharacterTextSplitter → smart chunking
  - HuggingFaceEmbeddings          → embeddings
  - PGVector (langchain_postgres)  → store in DB

Other features (search, hybrid) will be added in next steps.
"""

from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from config import DATABASE_URL, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

# ── Singletons — loaded once, reused forever ──────────────────
_embeddings  = None
_vectorstore = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load embedding model once at first call."""
    global _embeddings
    if _embeddings is None:
        print("Loading embedding model...")
        _embeddings = HuggingFaceEmbeddings(
            model_name    = EMBED_MODEL,
            model_kwargs  = {"device": "cpu"},
            encode_kwargs = {"normalize_embeddings": True},
        )
        print("✅ Embedding model ready")
    return _embeddings


def get_vectorstore() -> PGVector:
    """
    Connect to PGVector store.

    IMPORTANT connection URL rules:
      langchain_postgres needs psycopg v3 driver
      URL must be: postgresql+psycopg://  (NOT psycopg2://)
    """
    global _vectorstore
    if _vectorstore is None:
        # Convert standard postgresql:// to psycopg v3 format
        db_url = DATABASE_URL.replace(
            "postgresql://", "postgresql+psycopg://"
        )
        _vectorstore = PGVector(
            embeddings      = get_embeddings(),
            collection_name = "drupal_chunks",   # table name in pgvector
            connection      = db_url,
            use_jsonb       = True,              # better metadata filtering
        )
        print("✅ PGVector store connected")
    return _vectorstore


def store_node(node_id: int, title: str, content: str) -> int:
    """
    Store a Drupal node with LangChain chunking + embedding.

    Flow:
        1. Prepend title to content
        2. Split into chunks (RecursiveCharacterTextSplitter)
        3. Embed each chunk (HuggingFaceEmbeddings)
        4. Store all in pgvector (PGVector.add_documents)

    Returns: number of chunks stored
    """
    splitter    = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", ". ", " ", ""],
    )
    vectorstore = get_vectorstore()

    # Prepend title so every chunk has context
    full_text = f"{title}\n\n{content}"

    # Create LangChain Documents with metadata
    chunks = splitter.create_documents(
        texts     = [full_text],
        metadatas = [{
            "node_id": str(node_id),   # str — pgvector metadata is string
            "title":   title,
            "source":  f"drupal_node_{node_id}",
        }]
    )

    # Store all chunks — LangChain embeds + saves in one call
    vectorstore.add_documents(chunks)

    print(f"✅ Node {node_id} stored as {len(chunks)} chunks")
    return len(chunks)