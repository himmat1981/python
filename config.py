from dotenv import load_dotenv
import os

load_dotenv()

# ── Database ──────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@pgvector-db:5432/vectors"
)

# ── Groq LLM ──────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL    = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to .env")

# ── Embeddings ────────────────────────────────────────────────
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ── Spam Detection ────────────────────────────────────────────
MIN_MESSAGE_LENGTH  = int(os.getenv("MIN_MESSAGE_LENGTH", "3"))
MAX_MESSAGE_LENGTH  = int(os.getenv("MAX_MESSAGE_LENGTH", "500"))
REPEAT_CHAR_LIMIT   = int(os.getenv("REPEAT_CHAR_LIMIT", "10"))
GIBBERISH_THRESHOLD = float(os.getenv("GIBBERISH_THRESHOLD", "0.15"))

# ── Chunking ──────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "300"))    # words per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # overlap between chunks

# ── RAG ───────────────────────────────────────────────────────
RAG_TOP_K        = int(os.getenv("RAG_TOP_K", "10"))   # fetch top 10 for reranking
RAG_FINAL_TOP_K  = int(os.getenv("RAG_FINAL_TOP_K", "3"))  # return top 3 after rerank

# ── Response Cache ────────────────────────────────────────────
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1 hour