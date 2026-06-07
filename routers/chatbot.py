"""
routers/chatbot.py

Full LangChain RAG pipeline orchestrated by LangGraph:
  1. Check response cache     → return instantly if cached
  2. Spam check               → block or continue
  3. Intent classify          → tune search weights (factual/procedural/navigational/comparison/chitchat/off_topic)
  4. Retrieve (LangChain)     → intent_aware_retrieve (vector + keyword + intent-weighted rerank)
  5. Generate (LangChain)     → ChatPromptTemplate | ChatGroq | StrOutputParser
  6. Cache the answer         → store for 1 hour
"""

import asyncio

from fastapi import APIRouter, HTTPException
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, List, Optional

from models.schemas import ChatRequest
from services.spam import detect_spam
from services.retriever import intent_aware_retrieve
from services.intent_classifier import classify_intent
from services.llm import get_llm
from services.mlflow_tracker import track_chatbot, Timer
from db.vectors import (
    log_spam, get_spam_logs,
    get_response_cache, set_response_cache,
)
from config import LLM_MODEL

router = APIRouter(prefix="/chatbot", tags=["chatbot"])


# ── LangGraph State ───────────────────────────────────────────
class ChatState(TypedDict):
    question:      str
    intent:        Optional[str]
    intent_config: Optional[dict]
    context:       List[dict]
    answer:        str
    spam_reason:   Optional[str]
    document_id:   Optional[str]  # UUID from POST /documents/upload


# ══════════════════════════════════════════════════════════════
# NODE 1 — Spam Check
# ══════════════════════════════════════════════════════════════
def spam_check_node(state: ChatState) -> ChatState:
    reason = detect_spam(state["question"])
    return {**state, "spam_reason": reason}


# ══════════════════════════════════════════════════════════════
# NODE 2 — Classify intent to tune search strategy
# ══════════════════════════════════════════════════════════════
def intent_node(state: ChatState) -> ChatState:
    """
    Classifies the query into one of:
      factual / procedural / navigational / comparison / chitchat / off_topic

    Produces an intent_config dict that drives search weight tuning in
    the retrieve node. chitchat and off_topic skip retrieval entirely.
    """
    config = classify_intent(state["question"])
    return {**state, "intent": config["intent"], "intent_config": config}


# ══════════════════════════════════════════════════════════════
# NODE 3 — Retrieve via intent-aware hybrid search
# ══════════════════════════════════════════════════════════════
def retrieve_node(state: ChatState) -> ChatState:
    """
    Uses intent_aware_retrieve() instead of the generic HybridRetriever
    so that vector/keyword weights are tuned per intent:

      factual     → vector-heavy  (0.75 / 0.25)
      procedural  → balanced      (0.55 / 0.45)
      navigational → keyword-heavy (0.25 / 0.75)
      comparison  → balanced + wider fetch

    When document_id is present, results from the uploaded document are
    merged with the site knowledge base before reranking.

    chitchat / off_topic skip this node via routing (context stays empty).
    """
    docs = intent_aware_retrieve(
        state["question"],
        state["intent_config"],
        document_id=state.get("document_id"),
    )

    context = [
        {
            "node_id":      doc.metadata["node_id"],
            "title":        doc.metadata["title"],
            "content":      doc.page_content,
            "rerank_score": doc.metadata.get("rerank_score", 0.0),
            "source_type":  doc.metadata.get("source_type", "knowledge_base"),
        }
        for doc in docs
    ]
    return {**state, "context": context}


# ── LangChain RAG chain (built once at startup) ───────────────
_rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a helpful assistant for a Drupal website. "
            "Answer questions based only on the provided context. "
            "Context may include content from the site knowledge base and from "
            "user-uploaded documents — treat both equally. "
            "Mention the source title when referencing specific information. "
            "If the context doesn't contain relevant information, say so clearly."
        ),
    ),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

_rag_chain = _rag_prompt | get_llm() | StrOutputParser()


# ══════════════════════════════════════════════════════════════
# NODE 4 — Generate via LangChain LCEL chain
# ══════════════════════════════════════════════════════════════
def generate_node(state: ChatState) -> ChatState:
    """
    Formats retrieved context and runs the LangChain chain:
      ChatPromptTemplate | ChatGroq | StrOutputParser
    """
    context_text = "\n\n".join(
        f"[Source: {d['title']} | Relevance: {d.get('rerank_score', 0):.2f}]\n{d['content']}"
        for d in state["context"]
    ) if state["context"] else "No relevant content found."

    answer = _rag_chain.invoke({
        "context":  context_text,
        "question": state["question"],
    })
    return {**state, "answer": answer}


# ── Routing ───────────────────────────────────────────────────
def route_after_spam(state: ChatState) -> str:
    return "end" if state["spam_reason"] else "intent"


def route_after_intent(state: ChatState) -> str:
    """Skip retrieval for chitchat / off_topic — go straight to generate."""
    if state.get("intent_config", {}).get("skip_retrieval", False):
        return "generate"
    return "retrieve"


# ── Build LangGraph ───────────────────────────────────────────
def build_graph():
    graph = StateGraph(ChatState)
    graph.add_node("spam_check", spam_check_node)
    graph.add_node("intent",     intent_node)
    graph.add_node("retrieve",   retrieve_node)
    graph.add_node("generate",   generate_node)
    graph.set_entry_point("spam_check")
    graph.add_conditional_edges(
        "spam_check", route_after_spam,
        {"end": END, "intent": "intent"}
    )
    graph.add_conditional_edges(
        "intent", route_after_intent,
        {"retrieve": "retrieve", "generate": "generate"}
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
    Full LangChain RAG pipeline.

    Flow:
    1. Check response cache → instant return if found
    2. LangGraph runs: spam_check → retrieve → generate
    3. Cache answer for 1 hour
    """
    try:
        # ── Step 1: Check cache (skip when a document is attached) ──
        # Document-scoped answers must not be served from or written to the
        # general cache, since the same question with a different document
        # would produce a different answer.
        cached = None
        if not data.document_id:
            cached = await get_response_cache(data.question)
        if cached:
            return {
                "question": data.question,
                "answer":   cached["answer"],
                "sources":  cached["sources"],
                "cached":   True,
            }

        # ── Step 2–4: Run LangGraph pipeline ─────────────────
        # Run in a thread executor so the event loop stays free.
        # retrieve_node uses run_coroutine_threadsafe + future.result()
        # which requires it to run from a worker thread, not the event
        # loop thread — otherwise it deadlocks.
        loop = asyncio.get_running_loop()
        with Timer() as t:
            result = await loop.run_in_executor(None, rag_graph.invoke, {
                "question":      data.question,
                "intent":        None,
                "intent_config": None,
                "context":       [],
                "answer":        "",
                "spam_reason":   None,
                "document_id":   data.document_id,
            })

        # ── Handle spam ───────────────────────────────────────
        if result["spam_reason"]:
            await log_spam(data.question, result["spam_reason"])
            raise HTTPException(
                status_code=400,
                detail={
                    "error":   "spam_detected",
                    "reason":  result["spam_reason"],
                    "message": "Your message was flagged as spam."
                }
            )

        sources = [
            {
                "node_id":      d["node_id"],
                "title":        d["title"],
                "rerank_score": d.get("rerank_score", 0),
                "source_type":  d.get("source_type", "knowledge_base"),
            }
            for d in result["context"]
        ]

        # ── Step 5: Cache the answer (only for pure KB queries) ──
        if not data.document_id:
            await set_response_cache(data.question, result["answer"], sources)

        # ── Track in MLflow (background — don't block response) ──
        loop.run_in_executor(None, track_chatbot, data.question,
            result["answer"], result["context"], t.elapsed, LLM_MODEL, False, None)

        return {
            "question":    data.question,
            "intent":      result.get("intent"),
            "answer":      result["answer"],
            "sources":     sources,
            "document_id": data.document_id,
            "cached":      False,
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
