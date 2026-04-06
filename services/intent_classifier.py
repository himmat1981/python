"""
services/intent_classifier.py

Classifies user query intent to dynamically tune hybrid search behavior.

Intents and their effect on search:
  factual     → "What is X?", "Explain Y"       → vector-heavy  (0.75 / 0.25)
  procedural  → "How do I install X?"            → balanced      (0.55 / 0.45)
  navigational → "Find the page about X"         → keyword-heavy (0.25 / 0.75)
  comparison  → "Compare X and Y"               → balanced + wider top_k
  chitchat    → "Hi", "Thanks"                  → skip retrieval
  off_topic   → unrelated to site content       → skip retrieval

Flow:
  1. Quick rule-based pre-check (no LLM call for obvious chitchat)
  2. LLM call for all other queries (low max_tokens — fast)
  3. Return IntentConfig with weights and flags for the retriever
"""

import re
from services.llm import chat

# ── Intent → search configuration ─────────────────────────────
INTENT_CONFIGS = {
    "factual": {
        "vector_weight":   0.75,
        "keyword_weight":  0.25,
        "skip_retrieval":  False,
        "top_k_boost":     0,    # no extra results needed
    },
    "procedural": {
        "vector_weight":   0.55,
        "keyword_weight":  0.45,
        "skip_retrieval":  False,
        "top_k_boost":     0,
    },
    "navigational": {
        "vector_weight":   0.25,
        "keyword_weight":  0.75,
        "skip_retrieval":  False,
        "top_k_boost":     0,
    },
    "comparison": {
        "vector_weight":   0.60,
        "keyword_weight":  0.40,
        "skip_retrieval":  False,
        "top_k_boost":     2,    # fetch more results to compare across
    },
    "chitchat": {
        "vector_weight":   0.60,
        "keyword_weight":  0.40,
        "skip_retrieval":  True,
        "top_k_boost":     0,
    },
    "off_topic": {
        "vector_weight":   0.60,
        "keyword_weight":  0.40,
        "skip_retrieval":  True,
        "top_k_boost":     0,
    },
}

# Default fallback (if LLM returns something unexpected)
_DEFAULT_INTENT = "factual"

# ── Rule-based pre-check (no LLM cost for obvious cases) ──────
_CHITCHAT_PATTERNS = re.compile(
    r"^\s*(hi+|hello+|hey+|thanks?|thank you|bye+|good\s*(morning|evening|night)|how are you)\s*[!?.]*\s*$",
    re.IGNORECASE,
)


def _rule_based_intent(query: str) -> str | None:
    """
    Return an intent string for obviously simple queries, else None.
    Avoids an LLM call for short greetings/thanks.
    """
    if _CHITCHAT_PATTERNS.match(query):
        return "chitchat"
    return None


# ── LLM-based classifier ───────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a query intent classifier for a Drupal website chatbot.
Classify the user query into exactly ONE of these intents:

  factual     - asking for information or explanation ("What is X?", "Tell me about Y")
  procedural  - asking for steps or instructions ("How do I X?", "Steps to Y")
  navigational - looking for a specific page or link ("Find X", "Where is the page for Y")
  comparison  - comparing two or more things ("Compare X and Y", "Difference between X and Y")
  chitchat    - greetings, thanks, or small talk ("Hi", "Thanks", "How are you?")
  off_topic   - completely unrelated to the website or Drupal

Reply with ONLY the single intent word. No explanation, no punctuation.\
"""


def _llm_classify(query: str) -> str:
    """Call the LLM to classify intent. Returns a validated intent string."""
    response = chat(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": query},
        ],
        max_tokens=10,
        temperature=0.0,
    )
    intent = response.strip().lower()
    return intent if intent in INTENT_CONFIGS else _DEFAULT_INTENT


# ── Public API ─────────────────────────────────────────────────
def classify_intent(query: str) -> dict:
    """
    Classify the query intent and return a search configuration dict.

    Args:
        query: Raw user question

    Returns:
        dict with keys:
          intent          - one of the 6 intent labels
          vector_weight   - weight for vector similarity score
          keyword_weight  - weight for keyword overlap score
          skip_retrieval  - True → skip DB search entirely (chitchat/off_topic)
          top_k_boost     - extra results to fetch beyond RAG_TOP_K
    """
    intent = _rule_based_intent(query) or _llm_classify(query)
    config = INTENT_CONFIGS[intent].copy()
    config["intent"] = intent
    return config
