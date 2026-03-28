from fastapi import APIRouter, HTTPException
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional

from models.schemas import ChatRequest
from services.spam import detect_spam
from services.embeddings import encode
from services.llm import chat
from services.lora_model import lora_answer, check_lora_confidence
from services.mlflow_tracker import track_chatbot, Timer
from db.vectors import search_similar, log_spam, get_spam_logs
from config import LLM_MODEL

router = APIRouter(prefix="/chatbot", tags=["chatbot"])


# ── LangGraph State ───────────────────────────────────────────
class ChatState(TypedDict):
    question:        str
    context:         List[dict]
    answer:          str
    spam_reason:     Optional[str]
    lora_result:     dict          # stores lora answer + confidence
    final_source:    str           # "lora" or "rag"
    route_decision:  str           # "lora_accepted" or "rag_fallback"


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
# NODE 2 — LoRA Node (always called first)
# ══════════════════════════════════════════════════════════════
def lora_node(state: ChatState) -> ChatState:
    """
    Try LoRA trained model first.
    Returns answer + confidence score.
    If LoRA not trained yet — returns available: False.
    """
    result = lora_answer(state["question"])

    print(f"LoRA available: {result['available']}")
    if result["available"]:
        print(f"LoRA confidence: {result['confidence']}")
        print(f"LoRA answer preview: {result['answer'][:80]}")

    return {**state, "lora_result": result}


# ══════════════════════════════════════════════════════════════
# NODE 3 — Confidence Check
# Decides: accept LoRA answer or fallback to RAG?
# ══════════════════════════════════════════════════════════════
def confidence_check_node(state: ChatState) -> ChatState:
    """
    Check if LoRA answer is good enough.
    Three checks: confidence score, uncertainty phrases, answer length.
    """
    lora_result = state["lora_result"]
    is_good     = check_lora_confidence(lora_result)

    if is_good:
        # LoRA answer accepted — use it
        return {
            **state,
            "answer":         lora_result["answer"],
            "final_source":   "lora",
            "route_decision": "lora_accepted",
        }
    else:
        # LoRA failed — fallback to RAG
        return {
            **state,
            "route_decision": "rag_fallback",
            "final_source":   "rag",
        }


# ══════════════════════════════════════════════════════════════
# NODE 4 — RAG Retrieve
# ══════════════════════════════════════════════════════════════
def retrieve_node(state: ChatState) -> ChatState:
    """Search pgvector for relevant content."""
    vec  = encode(state["question"])
    docs = search_similar(vec, k=3)
    return {**state, "context": docs}


# ══════════════════════════════════════════════════════════════
# NODE 5 — RAG Generate
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
    return {
        **state,
        "answer":       answer,
        "final_source": "rag",
    }


# ══════════════════════════════════════════════════════════════
# ROUTING FUNCTIONS
# ══════════════════════════════════════════════════════════════
def route_after_spam(state: ChatState) -> str:
    """After spam check — blocked or continue to LoRA."""
    if state["spam_reason"]:
        return "end"
    return "lora"    # ← always try LoRA first


def route_after_confidence(state: ChatState) -> str:
    """After confidence check — accept LoRA or fallback to RAG."""
    return state["route_decision"]
    # "lora_accepted" → END
    # "rag_fallback"  → retrieve


# ══════════════════════════════════════════════════════════════
# BUILD LANGGRAPH
# ══════════════════════════════════════════════════════════════
def build_graph():
    graph = StateGraph(ChatState)

    # Register all nodes
    graph.add_node("spam_check",       spam_check_node)
    graph.add_node("lora",             lora_node)
    graph.add_node("confidence_check", confidence_check_node)
    graph.add_node("retrieve",         retrieve_node)
    graph.add_node("generate",         generate_node)

    # Entry point
    graph.set_entry_point("spam_check")

    # spam_check → blocked OR lora
    graph.add_conditional_edges(
        "spam_check",
        route_after_spam,
        {
            "end":  END,
            "lora": "lora"   # ← always goes to LoRA first
        }
    )

    # lora → confidence_check always
    graph.add_edge("lora", "confidence_check")

    # confidence_check → accept LoRA OR fallback to RAG
    graph.add_conditional_edges(
        "confidence_check",
        route_after_confidence,
        {
            "lora_accepted": END,        # LoRA was good → done
            "rag_fallback":  "retrieve"  # LoRA was bad → RAG
        }
    )

    # RAG pipeline
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
    Answer using LoRA first, RAG as fallback.

    Flow:
    1. Spam check
    2. LoRA tries to answer
    3. Confidence check — good enough?
    4. If yes → return LoRA answer
    5. If no  → search pgvector → Groq LLM
    """
    try:
        with Timer() as t:
            result = rag_graph.invoke({
                "question":       data.question,
                "context":        [],
                "answer":         "",
                "spam_reason":    None,
                "lora_result":    {},
                "final_source":   "",
                "route_decision": "",
            })

        # Track in MLflow
        track_chatbot(
            question      = data.question,
            answer        = result["answer"],
            sources       = result["context"],
            response_time = t.elapsed,
            model         = f"lora+{LLM_MODEL}",
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
            "question":     data.question,
            "answer":       result["answer"],
            "answered_by":  result["final_source"],   # "lora" or "rag"
            "confidence":   result["lora_result"].get("confidence", 0),
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


@router.get("/lora-status")
async def lora_status():
    """Check if LoRA model is trained and ready."""
    from services.lora_model import is_lora_trained
    import json
    import os

    trained = is_lora_trained()
    metadata = {}

    if trained:
        meta_path = "/app/lora_adapter/metadata.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)

    return {
        "lora_trained":   trained,
        "adapter_path":   "/app/lora_adapter",
        "base_model":     "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "metadata":       metadata,
        "train_command":  "docker exec -it vector-api python scripts/train_lora.py",
    }