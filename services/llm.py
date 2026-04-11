"""
services/llm.py

Unified LLM interface supporting multiple providers.
Switch by setting LLM_PROVIDER in .env:
  LLM_PROVIDER=groq    → uses Groq (default)
  LLM_PROVIDER=claude  → uses Anthropic Claude
"""

import logging
from langchain_core.messages import HumanMessage, SystemMessage
from config import LLM_PROVIDER, LLM_MODEL, GROQ_API_KEY, ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)

_llm = None


def get_llm(max_tokens: int = 1024, temperature: float = 0.3):
    """Return the LangChain LLM instance for the configured provider."""
    global _llm
    if _llm is None:
        if LLM_PROVIDER == "claude":
            from langchain_anthropic import ChatAnthropic
            _llm = ChatAnthropic(
                api_key=ANTHROPIC_API_KEY,
                model=LLM_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            logger.info("LLM provider: Claude (model=%s)", LLM_MODEL)
        else:
            from langchain_groq import ChatGroq
            _llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=LLM_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            logger.info("LLM provider: Groq (model=%s)", LLM_MODEL)
    return _llm


def chat(messages: list, max_tokens: int = 1024, temperature: float = 0.3) -> str:
    """
    Send a list of messages (OpenAI-style dicts) to the configured LLM.
    Converts dicts → LangChain message objects, invokes the LLM, returns text.
    """
    try:
        lc_messages = []
        for m in messages:
            if m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            else:
                lc_messages.append(HumanMessage(content=m["content"]))

        llm = get_llm(max_tokens=max_tokens, temperature=temperature)
        response = llm.invoke(lc_messages)
        return response.content.strip()

    except Exception as e:
        logger.error("LLM error (%s): %s", LLM_PROVIDER, e)
        return "LLM_ERROR"
