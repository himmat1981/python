"""
services/chunker.py

Smart text chunking with overlap.

Why chunking matters:
  Problem:  "2000 word article" → 1 embedding → loses detail
            Embedding can only capture overall topic, not specifics

  Solution: Split into 300-word chunks with 50-word overlap
            Each chunk gets its own embedding → much more precise search

Example:
  Article: "Drupal is a CMS... [2000 words about Drupal modules]"

  Without chunking:
    1 embedding for whole article
    Query "how to install views module" → embedding matches whole article weakly

  With chunking:
    chunk_0: "Drupal is a CMS... [300 words]"
    chunk_1: "[250 words overlap] ... modules overview [300 words]"
    chunk_2: "[250 words overlap] ... views module installation [300 words]"
    Query "how to install views module" → chunk_2 matches strongly ✅

Overlap prevents losing context at chunk boundaries.
"""

from typing import List
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping word chunks.

    Args:
        text:       Full content to chunk
        chunk_size: Words per chunk (default 300)
        overlap:    Words to repeat between chunks (default 50)

    Returns:
        List of text chunks
    """
    words = text.split()

    # If content is short enough — no chunking needed
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start  = 0

    while start < len(words):
        end   = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        # Move forward by (chunk_size - overlap) to create overlap
        start += chunk_size - overlap

        # Prevent infinite loop on last chunk
        if start >= len(words):
            break

    return chunks


def chunk_node(node_id: int, title: str, content: str) -> List[dict]:
    """
    Chunk a Drupal node into multiple pieces.

    Returns list of chunk dicts ready for DB storage.
    Each chunk includes title for context in search results.
    """
    chunks     = chunk_text(content)
    chunk_total = len(chunks)

    return [
        {
            "node_id":     node_id,
            "title":       title,
            "chunk_index": i,
            "chunk_total": chunk_total,
            # Prepend title to each chunk so context is never lost
            "content":     f"{title}\n\n{chunk}" if i == 0 else chunk,
        }
        for i, chunk in enumerate(chunks)
    ]