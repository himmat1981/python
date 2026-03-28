import psycopg2.extras
from typing import List
from db.connection import get_conn
from services.embeddings import encode  # your existing embedding service

def store_node_vector(node_id: int, title: str, content: str, embedding: list):
    """Save a Drupal node with its vector embedding."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO node_vectors (node_id, title, content, embedding)
                VALUES (%s, %s, %s, %s::vector)
                ON CONFLICT DO NOTHING;
            """, (node_id, title, content, embedding))
        conn.commit()


def search_similar(query_vector: list, k: int = 3) -> List[dict]:
    """Find top-k similar documents using cosine similarity."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                SELECT node_id, title, content,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM node_vectors
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_vector, query_vector, k))
            return [dict(row) for row in cur.fetchall()]


def log_spam(message: str, reason: str):
    """Save blocked spam messages for review."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO spam_log (message, reason) VALUES (%s, %s)",
                    (message, reason)
                )
            conn.commit()
    except Exception:
        pass  # never crash the app if spam logging fails


def get_spam_logs(limit: int = 100) -> List[dict]:
    """Retrieve recent spam log entries."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                SELECT id, message, reason, detected_at
                FROM spam_log
                ORDER BY detected_at DESC
                LIMIT %s;
            """, (limit,))
            return [dict(row) for row in cur.fetchall()]


def save_seo_cache(node_id: int, meta_title: str, meta_desc: str, keywords: str):
    """Cache generated SEO tags to avoid regenerating every time."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO seo_cache (node_id, meta_title, meta_desc, keywords)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (node_id)
                DO UPDATE SET
                    meta_title   = EXCLUDED.meta_title,
                    meta_desc    = EXCLUDED.meta_desc,
                    keywords     = EXCLUDED.keywords,
                    generated_at = CURRENT_TIMESTAMP;
            """, (node_id, meta_title, meta_desc, keywords))
        conn.commit()


def get_seo_cache(node_id: int):
    """Retrieve cached SEO tags for a node."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                SELECT meta_title, meta_desc, keywords, generated_at
                FROM seo_cache
                WHERE node_id = %s;
            """, (node_id,))
            row = cur.fetchone()
            return dict(row) if row else None
        
def search_similar_content(content: str, top_k: int = 3):

    # 🔥 Step 1: Convert text → embedding
    query_vector = encode(content)

    # 🔥 Step 2: Search similar
    results = search_similar(query_vector, top_k)

    return results