"""
main.py

App startup:
1. Init async DB pool
2. Create all tables
3. Pre-warm embedding model ← fixes Drupal timeout
4. Register all routers
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from db.connection import init_pool, close_pool, ensure_tables
from routers.nodes import router as nodes_router
from routers.chatbot import router as chatbot_router
from routers.documents import router as documents_router
from routers.seo import router as seo_router
from routers.nlp import router as nlp_router
from routers import governance
from routers.products import router as products_router
from routers.recommendations import router as recommendations_router


# ── Lifespan ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):

    # ── STARTUP ───────────────────────────────────────────────
    print("Starting Drupal AI API...")

    # 1. Init DB connection pool
    await init_pool()

    # 2. Create tables
    await ensure_tables()

    # 3. Pre-warm embedding model
    # Loads sentence-transformers into RAM at startup
    # Prevents Drupal timeout on first /nodes/store request
    print("Pre-warming embedding model...")
    from services.embeddings import get_embedder
    get_embedder()
    print("✅ Embedding model ready")

    print("✅ API ready to serve requests")

    yield

    # ── SHUTDOWN ──────────────────────────────────────────────
    await close_pool()
    print("API shutdown complete")


# ── Create app ────────────────────────────────────────────────
app = FastAPI(
    title       = "Drupal AI API",
    description = "RAG chatbot + SEO + spam detection for Drupal",
    version     = "3.0.0",
    lifespan    = lifespan,
)

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routers ───────────────────────────────────────────────────
app.include_router(nodes_router)
app.include_router(chatbot_router)
app.include_router(documents_router)
app.include_router(seo_router)
app.include_router(nlp_router)
app.include_router(governance.router)
app.include_router(products_router)
app.include_router(recommendations_router)


# ── Health check ──────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "status":  "Drupal AI API running 🚀",
        "version": "3.0.0",
        "endpoints": {
            "store":        "POST /nodes/store",
            "chatbot":      "POST /chatbot/ask",
            "seo_generate": "POST /seo/generate",
            "seo_cached":   "GET  /seo/cached/{node_id}",
            "spam_logs":    "GET  /chatbot/spam-logs",
            "cache_clear":  "DELETE /chatbot/cache/clear",
            "nlp_summary":  "POST /nlp/summarize",
            "nlp_moderate": "POST /nlp/moderate",
            "governance":   "POST /ai/governance",
            "docs":         "GET  /docs",
        }
    }