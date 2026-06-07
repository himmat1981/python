"""
db/vectors.py

All DB operations — async using asyncpg.

New features vs old file:
  - store_node_chunks()     → stores chunked embeddings instead of one big embedding
  - hybrid_search()         → vector + keyword search combined
  - get_response_cache()    → check if answer already cached
  - set_response_cache()    → save answer to cache
  - clean_expired_cache()   → remove old cache entries
"""

import asyncio
import json
import hashlib
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from db.connection import get_pool
from services.embeddings import encode, get_embedder
from config import CACHE_TTL_SECONDS, RAG_TOP_K


# ══════════════════════════════════════════════════════════════
# NODE CHUNKS — Smart chunking storage
# ══════════════════════════════════════════════════════════════

async def store_node_chunks(chunks: List[dict]):
    """
    Store all chunks of a node with their embeddings.
    Called from nodes router after chunking content.

    Each chunk dict must have:
        node_id, title, chunk_index, chunk_total, content
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        # Delete old chunks for this node first (in case of update)
        await conn.execute(
            "DELETE FROM node_chunks WHERE node_id = $1",
            chunks[0]["node_id"]
        )

        # Batch-embed in a thread executor — embed_documents() is CPU-bound
        # and would block the event loop if called directly in async code.
        texts      = [chunk["content"] for chunk in chunks]
        loop       = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None, get_embedder().embed_documents, texts
        )

        records = [
            (
                chunk["node_id"],
                chunk["title"],
                chunk["chunk_index"],
                chunk["chunk_total"],
                chunk["content"],
                str(embedding),
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        await conn.executemany("""
            INSERT INTO node_chunks
                (node_id, title, chunk_index, chunk_total, content, embedding)
            VALUES ($1, $2, $3, $4, $5, $6::vector)
        """, records)


async def hybrid_search(query: str, k: int = RAG_TOP_K) -> List[dict]:
    """
    Hybrid search = vector similarity + keyword (full-text) search combined.

    Why hybrid:
        Vector search:   finds semantically similar content
                         misses exact keyword matches sometimes
        Keyword search:  finds exact word matches
                         misses semantic meaning

        Combined:        best of both worlds

    Strategy:
        1. Vector search  → top-k by cosine similarity
        2. Keyword search → top-k by PostgreSQL full-text search
        3. Merge results  → deduplicate, keep highest score
    """
    pool         = get_pool()
    query_vector = encode(query)

    async with pool.acquire() as conn:
        async with conn.transaction():

            # Increase HNSW search candidate list for better recall (default is 40)
            await conn.execute("SET LOCAL hnsw.ef_search = 100")

            # ── Vector search ─────────────────────────────────────
            vector_rows = await conn.fetch("""
                SELECT
                    node_id, title, content, chunk_index,
                    1 - (embedding <=> $1::vector) AS similarity,
                    'vector' AS source
                FROM node_chunks
                ORDER BY embedding <=> $1::vector
                LIMIT $2;
            """, str(query_vector), k)

            # ── Keyword search (PostgreSQL full-text) ─────────────
            keyword_rows = await conn.fetch("""
                SELECT
                    node_id, title, content, chunk_index,
                    ts_rank(
                        to_tsvector('english', content),
                        plainto_tsquery('english', $1)
                    ) AS similarity,
                    'keyword' AS source
                FROM node_chunks
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
                ORDER BY similarity DESC
                LIMIT $2;
            """, query, k)

    # ── Merge and deduplicate ──────────────────────────────────
    seen    = {}
    results = []

    for row in list(vector_rows) + list(keyword_rows):
        # Use node_id + chunk_index as unique key
        key = f"{row['node_id']}_{row['chunk_index']}"
        r   = dict(row)

        if key not in seen:
            seen[key] = r
            results.append(r)
        else:
            # Keep higher similarity score
            if r["similarity"] > seen[key]["similarity"]:
                seen[key]["similarity"] = r["similarity"]
                seen[key]["source"]     = "hybrid"

    return results


# ══════════════════════════════════════════════════════════════
# RESPONSE CACHE — Cache chatbot answers
# ══════════════════════════════════════════════════════════════

def _make_cache_key(question: str) -> str:
    """
    Create a normalized cache key from question.
    Lowercase + strip + hash → consistent key for same question.

    "How do I install Drupal?" and "how do i install drupal?" → same key
    """
    normalized = question.lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()


async def get_response_cache(question: str) -> Optional[dict]:
    """
    Check if answer is cached and not expired.
    Returns cached response or None.
    """
    pool = get_pool()
    key  = _make_cache_key(question)

    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT question, answer, sources
            FROM response_cache
            WHERE question_key = $1
            AND expires_at > NOW();
        """, key)

        if row:
            return {
                "question": row["question"],
                "answer":   row["answer"],
                "sources":  json.loads(row["sources"]),
                "cached":   True,
            }
    return None


async def set_response_cache(question: str, answer: str, sources: list):
    """
    Save answer to cache with TTL expiry.
    Same question asked again within 1 hour → instant response.
    """
    pool       = get_pool()
    key        = _make_cache_key(question)
    expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=CACHE_TTL_SECONDS)

    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO response_cache
                (question_key, question, answer, sources, expires_at)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (question_key)
            DO UPDATE SET
                answer     = EXCLUDED.answer,
                sources    = EXCLUDED.sources,
                created_at = CURRENT_TIMESTAMP,
                expires_at = EXCLUDED.expires_at;
        """, key, question, answer, json.dumps(sources), expires_at)


async def clean_expired_cache():
    """Delete expired cache entries. Call periodically."""
    pool = get_pool()
    async with pool.acquire() as conn:
        deleted = await conn.execute(
            "DELETE FROM response_cache WHERE expires_at < NOW();"
        )
        print(f"Cache cleanup: {deleted}")


# ══════════════════════════════════════════════════════════════
# LEGACY — keep for backward compatibility
# ══════════════════════════════════════════════════════════════

async def store_node_vector(node_id: int, title: str, content: str, embedding: list):
    """Legacy single-vector storage. Use store_node_chunks() instead."""
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO node_vectors (node_id, title, content, embedding)
            VALUES ($1, $2, $3, $4::vector)
            ON CONFLICT DO NOTHING;
        """, node_id, title, content, str(embedding))


async def search_similar(query_vector: list, k: int = 3) -> List[dict]:
    """Legacy vector-only search. Use hybrid_search() instead."""
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT node_id, title, content,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM node_vectors
            ORDER BY embedding <=> $1::vector
            LIMIT $2;
        """, str(query_vector), k)
        return [dict(row) for row in rows]


async def search_similar_content(content: str, top_k: int = 3) -> List[dict]:
    """Search similar content by text — used by governance."""
    return await hybrid_search(content, k=top_k)


# ══════════════════════════════════════════════════════════════
# DOCUMENT CHUNKS — uploaded file storage and search
# ══════════════════════════════════════════════════════════════

async def store_document_chunks(document_id: str, filename: str, chunks: List[dict]):
    """
    Store all chunks of an uploaded document with their embeddings.

    Each chunk dict must have:
        content, chunk_index
    chunk_total is derived from len(chunks).
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        # Replace any existing chunks for this document_id
        await conn.execute(
            "DELETE FROM document_chunks WHERE document_id = $1",
            document_id
        )

        texts      = [chunk["content"] for chunk in chunks]
        loop       = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None, get_embedder().embed_documents, texts
        )

        chunk_total = len(chunks)
        records = [
            (
                document_id,
                filename,
                chunk["chunk_index"],
                chunk_total,
                chunk["content"],
                str(embedding),
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        await conn.executemany("""
            INSERT INTO document_chunks
                (document_id, filename, chunk_index, chunk_total, content, embedding)
            VALUES ($1, $2, $3, $4, $5, $6::vector)
        """, records)


async def document_hybrid_search(query: str, document_id: str, k: int = RAG_TOP_K) -> List[dict]:
    """
    Hybrid search scoped to a single uploaded document.

    Combines vector similarity and keyword full-text search,
    same strategy as hybrid_search() for node_chunks.
    Results are tagged with source_type='document' and include filename.
    """
    pool         = get_pool()
    query_vector = encode(query)

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("SET LOCAL hnsw.ef_search = 100")

            vector_rows = await conn.fetch("""
                SELECT
                    document_id, filename, content, chunk_index,
                    1 - (embedding <=> $1::vector) AS similarity,
                    'vector' AS source
                FROM document_chunks
                WHERE document_id = $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3;
            """, str(query_vector), document_id, k)

            keyword_rows = await conn.fetch("""
                SELECT
                    document_id, filename, content, chunk_index,
                    ts_rank(
                        to_tsvector('english', content),
                        plainto_tsquery('english', $1)
                    ) AS similarity,
                    'keyword' AS source
                FROM document_chunks
                WHERE document_id = $2
                  AND to_tsvector('english', content) @@ plainto_tsquery('english', $1)
                ORDER BY similarity DESC
                LIMIT $3;
            """, query, document_id, k)

    # Merge and deduplicate
    seen    = {}
    results = []

    for row in list(vector_rows) + list(keyword_rows):
        key = f"{row['document_id']}_{row['chunk_index']}"
        r   = dict(row)
        # Normalise to the same shape hybrid_search() returns
        r["node_id"]     = 0
        r["title"]       = row["filename"]
        r["source_type"] = "document"

        if key not in seen:
            seen[key] = r
            results.append(r)
        else:
            if r["similarity"] > seen[key]["similarity"]:
                seen[key]["similarity"] = r["similarity"]
                seen[key]["source"]     = "hybrid"

    return results


async def delete_document(document_id: str):
    """Delete all chunks for an uploaded document."""
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM document_chunks WHERE document_id = $1",
            document_id
        )


async def clean_expired_documents(max_age_hours: int = 24):
    """Delete document chunks older than max_age_hours. Call periodically."""
    pool = get_pool()
    async with pool.acquire() as conn:
        deleted = await conn.execute("""
            DELETE FROM document_chunks
            WHERE created_at < NOW() - ($1 * INTERVAL '1 hour');
        """, max_age_hours)
        print(f"Document cleanup: {deleted}")


# ══════════════════════════════════════════════════════════════
# SPAM LOG
# ══════════════════════════════════════════════════════════════

async def log_spam(message: str, reason: str):
    """Save blocked spam messages for review."""
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO spam_log (message, reason) VALUES ($1, $2)",
                message, reason
            )
    except Exception:
        pass


async def get_spam_logs(limit: int = 100) -> List[dict]:
    """Retrieve recent spam log entries."""
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, message, reason, detected_at
            FROM spam_log
            ORDER BY detected_at DESC
            LIMIT $1;
        """, limit)
        return [dict(row) for row in rows]


# ══════════════════════════════════════════════════════════════
# SEO CACHE
# ══════════════════════════════════════════════════════════════

async def save_seo_cache(node_id: int, meta_title: str, meta_desc: str, keywords: str):
    """Cache generated SEO tags."""
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO seo_cache (node_id, meta_title, meta_desc, keywords)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (node_id)
            DO UPDATE SET
                meta_title   = EXCLUDED.meta_title,
                meta_desc    = EXCLUDED.meta_desc,
                keywords     = EXCLUDED.keywords,
                generated_at = CURRENT_TIMESTAMP;
        """, node_id, meta_title, meta_desc, keywords)


async def get_seo_cache(node_id: int) -> Optional[dict]:
    """Retrieve cached SEO tags for a node."""
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT meta_title, meta_desc, keywords, generated_at
            FROM seo_cache WHERE node_id = $1;
        """, node_id)
        return dict(row) if row else None