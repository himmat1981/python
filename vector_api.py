from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from groq import Groq
from sentence_transformers import SentenceTransformer
import psycopg2
import psycopg2.extras
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional

import os
import re
import string
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@pgvector-db:5432/vectors")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL    = os.getenv("LLM_MODEL", "llama3-70b-8192")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to python/.env")

# ── Clients ───────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)

# Local embedding model (runs inside container, no API needed)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ── Spam Detection Config ─────────────────────────────────────────────────────

# Layer 1: message length limits
MIN_LENGTH = 3
MAX_LENGTH = 500

# Layer 2: repeated character limit
REPEAT_CHAR_LIMIT = 10

# Layer 3: gibberish detection word set
COMMON_ENGLISH_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come",
    "its", "over", "think", "also", "back", "after", "use", "two", "how",
    "our", "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "us", "is", "are", "was", "were",
    "has", "had", "did", "been", "being", "am", "does", "where", "why",
    "what", "which", "who", "whom", "whether",
    # Drupal specific words so they are never flagged as gibberish
    "drupal", "website", "content", "page", "node", "module", "theme",
    "install", "update", "delete", "create", "view", "edit", "user",
    "admin", "site", "field", "block", "menu", "role", "permission",
    "chatbot", "question", "answer", "help", "please", "thanks", "hello",
    "tell", "show", "find", "search", "list", "explain", "describe",
}
GIBBERISH_THRESHOLD = 0.15  # 15% — very lenient to avoid false positives

# Layer 4: prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore\s+(previous|prior|all|above|your)\s+instructions?",
    r"forget\s+(everything|all|your|previous|prior)",
    r"you\s+are\s+now\s+(a|an)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"act\s+as\s+(if\s+you\s+are|a|an)",
    r"do\s+not\s+follow",
    r"override\s+(your|the|all)\s+(instructions?|rules?|guidelines?)",
    r"system\s*:\s*you",
    r"disregard\s+(all|your|previous|prior)",
    r"new\s+instructions?\s*:",
    r"jailbreak",
    r"dan\s+mode",
]

# Layer 5: offensive / harmful words blocklist
OFFENSIVE_WORDS = {
    "spam", "scam", "fraud", "hack", "exploit",
}

# ── DB helpers ────────────────────────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(DATABASE_URL)

def ensure_table():
    """Create the vectors table and spam_log table if they don't exist."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS node_vectors (
                    id SERIAL PRIMARY KEY,
                    node_id INTEGER,
                    title TEXT,
                    content TEXT,
                    embedding vector(384)
                );
            """)
            # New: spam log table to store every blocked message
            cur.execute("""
                CREATE TABLE IF NOT EXISTS spam_log (
                    id SERIAL PRIMARY KEY,
                    message TEXT,
                    reason TEXT,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        conn.commit()

ensure_table()

def log_spam(message: str, reason: str):
    """Save blocked spam messages to DB for review and analysis."""
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

def search_similar(query: str, k: int = 3) -> List[dict]:
    """Find top-k similar documents using cosine similarity."""
    vec = embedder.encode(query).tolist()
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                SELECT node_id, title, content,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM node_vectors
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (vec, vec, k))
            return [dict(row) for row in cur.fetchall()]

# ── Spam Detection Functions ──────────────────────────────────────────────────

def check_length(message: str) -> Optional[str]:
    """Layer 1 — too short or too long."""
    if len(message.strip()) < MIN_LENGTH:
        return f"Message too short (minimum {MIN_LENGTH} characters)"
    if len(message) > MAX_LENGTH:
        return f"Message too long (maximum {MAX_LENGTH} characters)"
    return None

def check_repeated_chars(message: str) -> Optional[str]:
    """Layer 2 — same character repeated more than REPEAT_CHAR_LIMIT times."""
    pattern = r"(.)\1{" + str(REPEAT_CHAR_LIMIT) + r",}"
    if re.search(pattern, message):
        return "Message contains excessive repeated characters"
    return None

def check_prompt_injection(message: str) -> Optional[str]:
    """Layer 3 — attempts to override AI instructions."""
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, message.lower(), re.IGNORECASE):
            return "Message contains prompt injection attempt"
    return None

def check_offensive(message: str) -> Optional[str]:
    """Layer 4 — offensive or harmful words."""
    words_in_message = set(
        message.lower()
        .translate(str.maketrans("", "", string.punctuation))
        .split()
    )
    found = words_in_message.intersection(OFFENSIVE_WORDS)
    if found:
        return "Message contains offensive or harmful content"
    return None

def check_gibberish(message: str) -> Optional[str]:
    """Layer 5 — too few real English words (run last, most expensive)."""
    words = (
        message.lower()
        .translate(str.maketrans("", "", string.punctuation))
        .split()
    )
    if len(words) < 3:
        return None
    real_word_count = sum(1 for word in words if word in COMMON_ENGLISH_WORDS)
    ratio = real_word_count / len(words)
    if ratio < GIBBERISH_THRESHOLD:
        return f"Message appears to be gibberish ({int(ratio * 100)}% recognizable words)"
    return None

def detect_spam(message: str) -> Optional[str]:
    """
    Run all 5 spam checks in order.
    Returns the first reason found, or None if message is clean.
    """
    checks = [
        check_length,
        check_repeated_chars,
        check_prompt_injection,
        check_offensive,
        check_gibberish,
    ]
    for check in checks:
        reason = check(message)
        if reason:
            return reason
    return None

# ── LangGraph State ───────────────────────────────────────────────────────────
class ChatState(TypedDict):
    question:    str
    context:     List[dict]
    answer:      str
    spam_reason: Optional[str]  # None = clean, string = spam reason

# ── LangGraph Nodes ───────────────────────────────────────────────────────────

def spam_check_node(state: ChatState) -> ChatState:
    """Node 1 — Run spam detection before anything else."""
    reason = detect_spam(state["question"])
    if reason:
        log_spam(state["question"], reason)
    return {**state, "spam_reason": reason}

def retrieve_node(state: ChatState) -> ChatState:
    """Node 2 — Retrieve relevant Drupal content from pgvector."""
    docs = search_similar(state["question"], k=3)
    return {**state, "context": docs}

def generate_node(state: ChatState) -> ChatState:
    """Node 3 — Generate answer using Groq LLM."""
    context_text = "\n\n".join(
        f"Title: {d['title']}\n{d['content']}"
        for d in state["context"]
    ) if state["context"] else "No relevant content found."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant for a Drupal website. "
                "Answer questions based on the provided context. "
                "If the context doesn't contain relevant information, say so."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {state['question']}"
        }
    ]

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
    )
    answer = response.choices[0].message.content
    return {**state, "answer": answer}

# ── Router ────────────────────────────────────────────────────────────────────
def route_after_spam_check(state: ChatState) -> str:
    """
    Conditional edge after spam_check_node.
    Spam found → END (skip retrieve and generate).
    Clean      → continue to retrieve.
    """
    if state["spam_reason"]:
        return "end"
    return "retrieve"

# ── Build LangGraph ───────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("spam_check", spam_check_node)
    graph.add_node("retrieve",   retrieve_node)
    graph.add_node("generate",   generate_node)

    # Entry point is now spam_check (was retrieve before)
    graph.set_entry_point("spam_check")

    # Conditional edge — spam blocked or continue
    graph.add_conditional_edges(
        "spam_check",
        route_after_spam_check,
        {
            "end":      END,
            "retrieve": "retrieve"
        }
    )

    # These two edges are exactly the same as before
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

rag_graph = build_graph()

# ── Request/Response Models ───────────────────────────────────────────────────
class NodeData(BaseModel):
    node_id: int
    title: str
    content: str

class ChatRequest(BaseModel):
    question: str

    @field_validator("question")
    def must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()

class SearchQuery(BaseModel):
    query: str

# ── API Endpoints ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "LangGraph RAG API running 🚀",
        "llm": f"Groq / {LLM_MODEL}",
        "embeddings": "sentence-transformers/all-MiniLM-L6-v2",
        "graph": "spam_check → retrieve → generate",
        "spam_detection": "5 layers active",
    }

@app.post("/store")
async def store_node(data: NodeData):
    """Store a Drupal node with its vector embedding."""
    try:
        embedding = embedder.encode(data.content).tolist()
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO node_vectors (node_id, title, content, embedding)
                    VALUES (%s, %s, %s, %s::vector)
                    ON CONFLICT DO NOTHING;
                """, (data.node_id, data.title, data.content, embedding))
            conn.commit()
        return {"status": "stored successfully", "node_id": data.node_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage failed: {str(e)}")

@app.post("/chatbot")
async def chatbot(data: ChatRequest):
    """Answer a question using LangGraph RAG pipeline."""
    try:
        result = rag_graph.invoke({
            "question":    data.question,
            "context":     [],
            "answer":      "",
            "spam_reason": None,
        })

        # Spam detected — pipeline stopped early
        if result["spam_reason"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "error":   "spam_detected",
                    "reason":  result["spam_reason"],
                    "message": "Your message was flagged as spam. Please rephrase your question."
                }
            )

        return {
            "question": data.question,
            "answer":   result["answer"],
            "sources": [
                {"node_id": d["node_id"], "title": d["title"]}
                for d in result["context"]
            ],
        }
    except HTTPException:
        raise  # re-raise HTTP exceptions as-is
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

@app.post("/search")
async def search(query: SearchQuery):
    """Semantic search over stored Drupal content."""
    try:
        docs = search_similar(query.query, k=5)
        return {"results": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/spam-logs")
async def get_spam_logs():
    """View all blocked spam messages — useful for monitoring and tuning."""
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute("""
                    SELECT id, message, reason, detected_at
                    FROM spam_log
                    ORDER BY detected_at DESC
                    LIMIT 100;
                """)
                logs = [dict(row) for row in cur.fetchall()]
        return {"total": len(logs), "logs": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch logs: {str(e)}")