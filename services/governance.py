from db.vectors import search_similar_content
from services.crew_tasks import run_agents


def run_governance(content: str):

    # 🔥 RAG context
    rag_results = search_similar_content(content, top_k=3)
    rag_context = "\n".join([r["content"] for r in rag_results])

    return run_agents(content, rag_context)