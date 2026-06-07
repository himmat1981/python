"""
db/connection.py

Async connection pool using asyncpg.
Includes all tables: node_vectors, chunks, response_cache, spam_log, seo_cache.
"""

import asyncio
import asyncpg
from config import DATABASE_URL

_pool: asyncpg.Pool = None
_main_loop: asyncio.AbstractEventLoop = None


def get_main_loop() -> asyncio.AbstractEventLoop:
    """Return the main event loop captured at startup."""
    return _main_loop


async def init_pool():
    """Initialize async connection pool at app startup."""
    global _pool, _main_loop
    _main_loop = asyncio.get_running_loop()
    if _pool is None:
        db_url = DATABASE_URL.replace("postgresql://", "postgres://")
        _pool = await asyncpg.create_pool(
            dsn      = db_url,
            min_size = 2,
            max_size = 10,
        )
        print("✅ Async DB pool initialized (min=2, max=10)")


async def close_pool():
    """Close all connections at app shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        print("DB pool closed")


def get_pool() -> asyncpg.Pool:
    """Return pool — used by all DB functions."""
    return _pool


async def ensure_tables():
    """Create all required tables on startup."""
    pool = get_pool()
    async with pool.acquire() as conn:

        # pgvector extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # ── Node vectors (full content — kept for backward compat) ──
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS node_vectors (
                id        SERIAL PRIMARY KEY,
                node_id   INTEGER,
                title     TEXT,
                content   TEXT,
                embedding vector(384)
            );
        """)

        # ── Node chunks (NEW — smart chunking) ──────────────────────
        # Each node is split into multiple smaller chunks
        # chunk_index = which chunk (0, 1, 2...)
        # chunk_total = how many chunks this node has
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS node_chunks (
                id          SERIAL PRIMARY KEY,
                node_id     INTEGER NOT NULL,
                title       TEXT,
                chunk_index INTEGER NOT NULL,
                chunk_total INTEGER NOT NULL,
                content     TEXT,
                embedding   vector(384),
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # HNSW index for fast approximate nearest-neighbour search.
        # m=16: connections per layer (higher → better recall, more RAM)
        # ef_construction=64: build-time candidate list size (higher → better quality index)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding
            ON node_chunks USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)

        # Index for fast full-text keyword search on chunks
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_content_fts
            ON node_chunks USING gin(to_tsvector('english', content));
        """)

        # ── Response cache (NEW — cache chatbot answers) ─────────────
        # Stores answers for 1 hour — repeat questions return instantly
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS response_cache (
                id           SERIAL PRIMARY KEY,
                question_key TEXT UNIQUE NOT NULL,
                question     TEXT,
                answer       TEXT,
                sources      TEXT,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at   TIMESTAMP NOT NULL
            );
        """)

        # Index for fast cache lookup
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_response_cache_key
            ON response_cache (question_key);
        """)

        # ── Spam log ──────────────────────────────────────────────────
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS spam_log (
                id          SERIAL PRIMARY KEY,
                message     TEXT,
                reason      TEXT,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # ── SEO cache ─────────────────────────────────────────────────
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS seo_cache (
                id           SERIAL PRIMARY KEY,
                node_id      INTEGER UNIQUE,
                meta_title   TEXT,
                meta_desc    TEXT,
                keywords     TEXT,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # ── Summary cache ─────────────────────────────────────────────
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS summary_cache (
                id           SERIAL PRIMARY KEY,
                node_id      INTEGER UNIQUE,
                summary      TEXT,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # ── Products (ecommerce) ───────────────────────────────────────
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id          SERIAL PRIMARY KEY,
                product_id  INTEGER UNIQUE NOT NULL,
                title       TEXT NOT NULL,
                description TEXT DEFAULT '',
                price       DECIMAL(10,2) DEFAULT 0,
                category    TEXT DEFAULT '',
                sku         TEXT DEFAULT '',
                embedding   vector(384),
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # HNSW index for fast product similarity search
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_products_embedding
            ON products USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)

        # ── User interactions (ecommerce) ──────────────────────────────
        # Tracks views, clicks, add-to-cart, purchases per user+product
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_interactions (
                id               SERIAL PRIMARY KEY,
                user_id          INTEGER NOT NULL,
                product_id       INTEGER NOT NULL,
                interaction_type TEXT NOT NULL,
                score            DECIMAL(5,2) DEFAULT 1.0,
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_interactions_user
            ON user_interactions (user_id, created_at DESC);
        """)

        # ── Recommendation cache (ecommerce) ──────────────────────────
        # Caches per-user recommendations for 30 minutes
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS recommendation_cache (
                id              SERIAL PRIMARY KEY,
                user_id         INTEGER UNIQUE NOT NULL,
                recommendations TEXT NOT NULL,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at      TIMESTAMP NOT NULL
            );
        """)

        # ── Document chunks (uploaded files) ───────────────────────
        # Stores chunks from user-uploaded documents (PDF, DOCX, TXT).
        # document_id is a UUID returned to the caller on upload.
        # Rows are auto-cleaned after 24 hours via clean_expired_documents().
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id           SERIAL PRIMARY KEY,
                document_id  TEXT NOT NULL,
                filename     TEXT NOT NULL,
                chunk_index  INTEGER NOT NULL,
                chunk_total  INTEGER NOT NULL,
                content      TEXT,
                embedding    vector(384),
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # HNSW index for fast ANN search on uploaded document chunks
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
            ON document_chunks USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)

        # Full-text search index on document chunk content
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_fts
            ON document_chunks USING gin(to_tsvector('english', content));
        """)

        # Lookup by document_id (used in search + delete)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_id
            ON document_chunks (document_id);
        """)

    print("✅ DB tables verified")