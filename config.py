from dotenv import load_dotenv
import os

load_dotenv()

# ── Database ──────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@pgvector-db:5432/vectors"
)

# ── LLM Provider ──────────────────────────────────────────────
# Set LLM_PROVIDER=groq or LLM_PROVIDER=claude in .env
LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_MODEL       = os.getenv("LLM_MODEL", "llama3-70b-8192")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to .env")
if LLM_PROVIDER == "claude" and not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env")

# ── Embeddings ────────────────────────────────────────────────
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ── Spam Detection ────────────────────────────────────────────
MIN_MESSAGE_LENGTH  = int(os.getenv("MIN_MESSAGE_LENGTH", "3"))
MAX_MESSAGE_LENGTH  = int(os.getenv("MAX_MESSAGE_LENGTH", "500"))
REPEAT_CHAR_LIMIT   = int(os.getenv("REPEAT_CHAR_LIMIT", "10"))
GIBBERISH_THRESHOLD = float(os.getenv("GIBBERISH_THRESHOLD", "0.15"))

# ── Chunking ──────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ── RAG ───────────────────────────────────────────────────────
RAG_TOP_K       = int(os.getenv("RAG_TOP_K", "10"))
RAG_FINAL_TOP_K = int(os.getenv("RAG_FINAL_TOP_K", "3"))

# ── Response Cache ────────────────────────────────────────────
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))