from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db.connection import ensure_tables
from routers.nodes import router as nodes_router
from routers.chatbot import router as chatbot_router
from routers.seo import router as seo_router
from routers.nlp import router as nlp_router
from routers import governance



# ── Create app ────────────────────────────────────────────────
app = FastAPI(
    title="Drupal AI API",
    description="RAG chatbot + SEO generation + spam detection for Drupal",
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── DB setup on startup ───────────────────────────────────────
@app.on_event("startup")
def on_startup():
    ensure_tables()

# ── Register routers ──────────────────────────────────────────
app.include_router(nodes_router)
app.include_router(chatbot_router)
app.include_router(seo_router)
app.include_router(nlp_router)
app.include_router(governance.router)
# ── Health check ──────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status":   "Drupal AI API running 🚀",
        "version":  "1.0.0",
        "endpoints": {
            "store":        "POST /nodes/store",
            "chatbot":      "POST /chatbot/ask",
            "seo_generate": "POST /seo/generate",
            "seo_cached":   "GET  /seo/cached/{node_id}",
            "spam_logs":    "GET  /chatbot/spam-logs",
            "docs":         "GET  /docs",
        }
    }