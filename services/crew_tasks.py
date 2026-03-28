from services.crew_agents import (
    quality_agent,
    fact_checker_agent,
    rewrite_agent,
    compliance_agent
)


def run_agents(content: str, rag_context: str):

    quality = quality_agent(content)

    fact_check = fact_checker_agent(content, rag_context)

    improved = rewrite_agent(content)

    compliance = compliance_agent(content)

    return {
        "quality": quality,
        "fact_check": fact_check,
        "improved_content": improved.get("improved_content") if isinstance(improved, dict) else improved,
        "compliance": compliance
    }