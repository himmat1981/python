from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from config import GROQ_API_KEY, LLM_MODEL

# Singleton LangChain ChatGroq client
_llm = None


def get_llm(max_tokens: int = 1024, temperature: float = 0.3) -> ChatGroq:
    """Return the ChatGroq LangChain LLM instance."""
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=LLM_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    return _llm


def chat(messages: list, max_tokens: int = 1024, temperature: float = 0.3) -> str:
    """
    Send a list of messages (OpenAI-style dicts) to Groq via LangChain.
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
        print(f"LLM Error: {e}")
        return "LLM_ERROR"
