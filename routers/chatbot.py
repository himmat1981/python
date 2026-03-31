"""
routers/chatbot.py

Full optimized RAG pipeline:
  1. Check response cache → return instantly if cached
  2. Spam check
  3. Hybrid search (vector + keyword)
  4. Rerank results
  5. Generate answer via Groq
  6. Cache the answer for next time
"""

from fastapi import APIRouter, HTTPException
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
import asyncio

from models.schemas import ChatRequest
from services.spam import detect_spam
from services.reranker import rerank
from services.llm import chat
from services.mlflow_tracker import track_chatbot, Timer
from db.vectors import (
    hybrid_search,
    log_spam, get_spam_logs,
    get_response_cache, set_response_cache,
)
from config import LLM_MODEL, RAG_FINAL_TOP_K

router = APIRouter(prefix="/chatbot", tags=["chatbot"])


# ── LangGraph State ───────────────────────────────────────────
class ChatState(TypedDict):
    question:    str
    context:     List[dict]
    answer:      str
    spam_reason: Optional[str]


# ══════════════════════════════════════════════════════════════
# NODE 1 — Spam Check
# ══════════════════════════════════════════════════════════════
def spam_check_node(state: ChatState) -> ChatState:
    reason = detect_spam(state["question"])
    return {**state, "spam_reason": reason}


# ══════════════════════════════════════════════════════════════
# NODE 2 — Hybrid Search + Rerank
# ══════════════════════════════════════════════════════════════
def retrieve_node(state: ChatState) -> ChatState:
    """
    Hybrid search: vector + keyword combined.
    Then rerank results for best relevance.
    """
    # Run async search in sync LangGraph node
    loop = asyncio.get_event_loop()
    docs = loop.run_until_complete(hybrid_search(state["question"]))

    # Rerank: from top-10 → best 3
    reranked = rerank(state["question"], docs, top_k=RAG_FINAL_TOP_K)

    return {**state, "context": reranked}


# ══════════════════════════════════════════════════════════════
# NODE 3 — Generate
# ══════════════════════════════════════════════════════════════
def generate_node(state: ChatState) -> ChatState:
    """Generate answer using Groq LLM with reranked context."""
    context_text = "\n\n".join(
        f"[Source: {d['title']} | Relevance: {d.get('rerank_score', 0):.2f}]\n{d['content']}"
        for d in state["context"]
    ) if state["context"] else "No relevant content found."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant for a Drupal website. "
                "Answer questions based only on the provided context. "
                "Mention the source title when referencing specific information. "
                "If the context doesn't contain relevant information, say so clearly."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {state['question']}"
        }
    ]
    answer = chat(messages)
    return {**state, "answer": answer}


# ── Routing ───────────────────────────────────────────────────
def route_after_spam(state: ChatState) -> str:
    return "end" if state["spam_reason"] else "retrieve"


# ── Build Graph ───────────────────────────────────────────────
def build_graph():
    graph = StateGraph(ChatState)
    graph.add_node("spam_check", spam_check_node)
    graph.add_node("retrieve",   retrieve_node)
    graph.add_node("generate",   generate_node)
    graph.set_entry_point("spam_check")
    graph.add_conditional_edges(
        "spam_check", route_after_spam,
        {"end": END, "retrieve": "retrieve"}
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
    Full optimized RAG pipeline.

    Flow:
    1. Check response cache → instant return if found
    2. Spam check
    3. Hybrid search (vector + keyword) → top 10
    4. Rerank → top 3
    5. Groq LLM generates answer
    6. Cache answer for 1 hour
    """
    try:
        # ── Step 1: Check cache ───────────────────────────────
        cached = await get_response_cache(data.question)
        if cached:
            return {
                "question": data.question,
                "answer":   cached["answer"],
                "sources":  cached["sources"],
                "cached":   True,
            }

        # ── Step 2-5: Run RAG pipeline ────────────────────────
        with Timer() as t:
            result = rag_graph.invoke({
                "question":    data.question,
                "context":     [],
                "answer":      "",
                "spam_reason": None,
            })

        # ── Handle spam ───────────────────────────────────────
        if result["spam_reason"]:
            await log_spam(data.question, result["spam_reason"])
            raise HTTPException(
                status_code = 400,
                detail = {
                    "error":   "spam_detected",
                    "reason":  result["spam_reason"],
                    "message": "Your message was flagged as spam."
                }
            )

        # Build sources list
        sources = [
            {
                "node_id":      d["node_id"],
                "title":        d["title"],
                "rerank_score": d.get("rerank_score", 0),
            }
            for d in result["context"]
        ]

        # ── Step 6: Cache the answer ──────────────────────────
        await set_response_cache(data.question, result["answer"], sources)

        # ── Track in MLflow ───────────────────────────────────
        track_chatbot(
            question      = data.question,
            answer        = result["answer"],
            sources       = result["context"],
            response_time = t.elapsed,
            model         = LLM_MODEL,
            spam_detected = False,
            spam_reason   = None,
        )

        return {
            "question": data.question,
            "answer":   result["answer"],
            "sources":  sources,
            "cached":   False,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/spam-logs")
async def spam_logs():
    """View all blocked spam messages."""
    try:
        logs = await get_spam_logs(limit=100)
        return {"total": len(logs), "logs": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache/clear")
async def clear_cache():
    """Clear expired cache entries."""
    from db.vectors import clean_expired_cache
    await clean_expired_cache()
    return {"status": "cache cleaned"}