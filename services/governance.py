"""
services/governance.py

Runs multi-agent governance pipeline on content.
Uses async DB search for RAG context.
"""

from db.vectors import search_similar_content
from services.crew_tasks import run_agents


async def run_governance(content: str):
    """
    Fetch similar content from pgvector (async)
    then run quality, fact-check, rewrite, compliance agents.
    """
    rag_results = await search_similar_content(content, top_k=3)
    rag_context = "\n".join([r["content"] for r in rag_results])
    return run_agents(content, rag_context)