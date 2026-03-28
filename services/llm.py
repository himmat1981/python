from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL

# Singleton — one client instance reused for all requests
_client = None

def get_client() -> Groq:
    """Return the Groq client, creating it if not already created."""
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def chat(messages: list, max_tokens: int = 1024, temperature: float = 0.3) -> str:
    try:
        response = get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return "LLM_ERROR"