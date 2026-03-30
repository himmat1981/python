"""
routers/chatbot.py

Clean RAG-only chatbot using LangGraph.
LoRA removed — all answers via pgvector + Groq LLM.
"""

from fastapi import APIRouter, HTTPException
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional

from models.schemas import ChatRequest
from services.spam import detect_spam
from services.embeddings import encode
from services.llm import chat
from services.mlflow_tracker import track_chatbot, Timer
from db.vectors import search_similar, log_spam, get_spam_logs
from config import LLM_MODEL

router = APIRouter(prefix="/chatbot", tags=["chatbot"])


# ── LangGraph State ───────────────────────────────────────────
class ChatState(TypedDict):
    question:     str
    context:      List[dict]
    answer:       str
    spam_reason:  Optional[str]


# ══════════════════════════════════════════════════════════════
# NODE 1 — Spam Check
# ══════════════════════════════════════════════════════════════
def spam_check_node(state: ChatState) -> ChatState:
    """Check for spam before doing anything else."""
    reason = detect_spam(state["question"])
    if reason:
        log_spam(state["question"], reason)
    return {**state, "spam_reason": reason}


# ══════════════════════════════════════════════════════════════
# NODE 2 — RAG Retrieve
# ══════════════════════════════════════════════════════════════
def retrieve_node(state: ChatState) -> ChatState:
    """Search pgvector for relevant content."""
    vec  = encode(state["question"])
    docs = search_similar(vec, k=3)
    return {**state, "context": docs}


# ══════════════════════════════════════════════════════════════
# NODE 3 — RAG Generate
# ══════════════════════════════════════════════════════════════
def generate_node(state: ChatState) -> ChatState:
    """Generate answer using Groq LLM with retrieved context."""
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
    answer = chat(messages)
    return {**state, "answer": answer}


# ══════════════════════════════════════════════════════════════
# ROUTING FUNCTIONS
# ══════════════════════════════════════════════════════════════
def route_after_spam(state: ChatState) -> str:
    """After spam check — blocked or continue to RAG."""
    if state["spam_reason"]:
        return "end"
    return "retrieve"


# ══════════════════════════════════════════════════════════════
# BUILD LANGGRAPH
# ══════════════════════════════════════════════════════════════
def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("spam_check", spam_check_node)
    graph.add_node("retrieve",   retrieve_node)
    graph.add_node("generate",   generate_node)

    graph.set_entry_point("spam_check")

    graph.add_conditional_edges(
        "spam_check",
        route_after_spam,
        {
            "end":      END,
            "retrieve": "retrieve"
        }
    )

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


rag_graph = build_graph()


# ══════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════
@router.post("/ask")
async def ask(data: ChatRequest):
    """
    Answer questions using RAG pipeline.

    Flow:
    1. Spam check
    2. Search pgvector for relevant content
    3. Groq LLM generates answer from context
    """
    try:
        with Timer() as t:
            result = rag_graph.invoke({
                "question":    data.question,
                "context":     [],
                "answer":      "",
                "spam_reason": None,
            })

        # Track in MLflow
        track_chatbot(
            question      = data.question,
            answer        = result["answer"],
            sources       = result["context"],
            response_time = t.elapsed,
            model         = LLM_MODEL,
            spam_detected = bool(result["spam_reason"]),
            spam_reason   = result["spam_reason"],
        )

        # Spam blocked
        if result["spam_reason"]:
            raise HTTPException(
                status_code = 400,
                detail = {
                    "error":   "spam_detected",
                    "reason":  result["spam_reason"],
                    "message": "Your message was flagged as spam."
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
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/spam-logs")
async def spam_logs():
    """View all blocked spam messages."""
    try:
        logs = get_spam_logs(limit=100)
        return {"total": len(logs), "logs": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))