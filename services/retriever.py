"""
services/retriever.py

LangChain BaseRetriever wrapping hybrid search + reranker.

Flow:
  1. hybrid_search()  → vector + keyword combined (top-10)
  2. rerank()         → scored by keyword overlap + similarity (top-3)
  3. Returns List[Document] — standard LangChain format

Also exposes intent_aware_retrieve() for the LangGraph chatbot pipeline,
which accepts an intent config dict (from intent_classifier) to tune weights.
"""

import asyncio
from typing import List

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from db.vectors import hybrid_search
from db.connection import get_main_loop
from services.reranker import rerank
from config import RAG_FINAL_TOP_K, RAG_TOP_K


class HybridRetriever(BaseRetriever):
    """
    LangChain retriever using hybrid vector+keyword search with reranking.
    Compatible with any LangChain chain via .invoke() or .get_relevant_documents().
    """
    top_k: int = RAG_FINAL_TOP_K

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        # hybrid_search is async and uses an asyncpg pool bound to the main
        # event loop. Running it in a new loop causes "attached to a different
        # loop" errors. Schedule it on the main loop and block until done.
        main_loop = get_main_loop()
        future = asyncio.run_coroutine_threadsafe(hybrid_search(query), main_loop)
        raw_docs = future.result(timeout=30)

        reranked = rerank(query, raw_docs, top_k=self.top_k)

        return [
            Document(
                page_content=d["content"],
                metadata={
                    "node_id":      d["node_id"],
                    "title":        d["title"],
                    "rerank_score": d.get("rerank_score", 0.0),
                },
            )
            for d in reranked
        ]


# Singleton
_retriever: HybridRetriever = None


def get_retriever() -> HybridRetriever:
    """Return the HybridRetriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def intent_aware_retrieve(query: str, intent_config: dict) -> List[Document]:
    """
    Intent-aware retrieval for the LangGraph pipeline.

    Unlike the generic HybridRetriever, this uses the intent_config produced
    by intent_classifier.classify_intent() to tune reranking weights and top_k.

    Args:
        query:         User question
        intent_config: Dict from classify_intent() — must contain:
                         vector_weight, keyword_weight, top_k_boost

    Returns:
        List[Document] — same format as HybridRetriever for drop-in use
    """
    vector_weight  = intent_config.get("vector_weight",  0.6)
    keyword_weight = intent_config.get("keyword_weight", 0.4)
    top_k_boost    = intent_config.get("top_k_boost",    0)
    fetch_k        = RAG_TOP_K + top_k_boost

    main_loop = get_main_loop()
    future    = asyncio.run_coroutine_threadsafe(
        hybrid_search(query, k=fetch_k), main_loop
    )
    raw_docs = future.result(timeout=30)

    reranked = rerank(
        query,
        raw_docs,
        top_k=RAG_FINAL_TOP_K,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )

    return [
        Document(
            page_content=d["content"],
            metadata={
                "node_id":      d["node_id"],
                "title":        d["title"],
                "rerank_score": d.get("rerank_score", 0.0),
            },
        )
        for d in reranked
    ]
