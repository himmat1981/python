import psycopg2
import psycopg2.extras
from config import DATABASE_URL


def get_conn():
    """Create a fresh database connection."""
    return psycopg2.connect(DATABASE_URL)


def ensure_tables():
    """Create all required tables on startup if they don't exist."""
    with get_conn() as conn:
        with conn.cursor() as cur:

            # pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Drupal node vectors
            cur.execute("""
                CREATE TABLE IF NOT EXISTS node_vectors (
                    id          SERIAL PRIMARY KEY,
                    node_id     INTEGER,
                    title       TEXT,
                    content     TEXT,
                    embedding   vector(384)
                );
            """)

            # Spam log
            cur.execute("""
                CREATE TABLE IF NOT EXISTS spam_log (
                    id          SERIAL PRIMARY KEY,
                    message     TEXT,
                    reason      TEXT,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # SEO cache
            cur.execute("""
                CREATE TABLE IF NOT EXISTS seo_cache (
                    id           SERIAL PRIMARY KEY,
                    node_id      INTEGER UNIQUE,
                    meta_title   TEXT,
                    meta_desc    TEXT,
                    keywords     TEXT,
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # ── NEW: Summary cache ────────────────────────────
            # Stores AI generated summaries
            # Same node never summarized twice — zero cost on repeat
            cur.execute("""
                CREATE TABLE IF NOT EXISTS summary_cache (
                    id           SERIAL PRIMARY KEY,
                    node_id      INTEGER UNIQUE,
                    summary      TEXT,
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # ── NEW: LoRA training log ────────────────────────
            # Tracks when LoRA was trained and on how many nodes
            # Useful for knowing when to retrain
            cur.execute("""
                CREATE TABLE IF NOT EXISTS lora_training_log (
                    id            SERIAL PRIMARY KEY,
                    nodes_trained INTEGER,
                    base_model    TEXT,
                    adapter_path  TEXT,
                    trained_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

        conn.commit()