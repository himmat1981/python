import re
import string
from typing import Optional
from config import MIN_MESSAGE_LENGTH, MAX_MESSAGE_LENGTH, REPEAT_CHAR_LIMIT, GIBBERISH_THRESHOLD

# ── Word set for gibberish detection ─────────────────────────
COMMON_ENGLISH_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come",
    "its", "over", "think", "also", "back", "after", "use", "two", "how",
    "our", "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "us", "is", "are", "was", "were",
    "has", "had", "did", "been", "being", "am", "does", "where", "why",
    "drupal", "website", "content", "page", "node", "module", "theme",
    "install", "update", "delete", "create", "view", "edit", "user",
    "admin", "site", "field", "block", "menu", "role", "permission",
    "chatbot", "question", "answer", "help", "please", "thanks", "hello",
    "tell", "show", "find", "search", "list", "explain", "describe",
}

# ── Prompt injection patterns ─────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore\s+(previous|prior|all|above|your)\s+instructions?",
    r"forget\s+(everything|all|your|previous|prior)",
    r"you\s+are\s+now\s+(a|an)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"act\s+as\s+(if\s+you\s+are|a|an)",
    r"do\s+not\s+follow",
    r"override\s+(your|the|all)\s+(instructions?|rules?|guidelines?)",
    r"system\s*:\s*you",
    r"disregard\s+(all|your|previous|prior)",
    r"new\s+instructions?\s*:",
    r"jailbreak",
    r"dan\s+mode",
]

# ── Offensive words ───────────────────────────────────────────
OFFENSIVE_WORDS = {
    "spam", "scam", "fraud", "hack", "exploit",
}


def check_length(message: str) -> Optional[str]:
    if len(message.strip()) < MIN_MESSAGE_LENGTH:
        return f"Message too short (minimum {MIN_MESSAGE_LENGTH} characters)"
    if len(message) > MAX_MESSAGE_LENGTH:
        return f"Message too long (maximum {MAX_MESSAGE_LENGTH} characters)"
    return None


def check_repeated_chars(message: str) -> Optional[str]:
    pattern = r"(.)\1{" + str(REPEAT_CHAR_LIMIT) + r",}"
    if re.search(pattern, message):
        return "Message contains excessive repeated characters"
    return None


def check_prompt_injection(message: str) -> Optional[str]:
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, message.lower(), re.IGNORECASE):
            return "Message contains prompt injection attempt"
    return None


def check_offensive(message: str) -> Optional[str]:
    words = set(
        message.lower()
        .translate(str.maketrans("", "", string.punctuation))
        .split()
    )
    if words.intersection(OFFENSIVE_WORDS):
        return "Message contains offensive or harmful content"
    return None


def check_gibberish(message: str) -> Optional[str]:
    words = (
        message.lower()
        .translate(str.maketrans("", "", string.punctuation))
        .split()
    )
    if len(words) < 3:
        return None
    ratio = sum(1 for w in words if w in COMMON_ENGLISH_WORDS) / len(words)
    if ratio < GIBBERISH_THRESHOLD:
        return f"Message appears to be gibberish ({int(ratio * 100)}% recognizable words)"
    return None


def detect_spam(message: str) -> Optional[str]:
    """
    Run all 5 checks in order — cheapest first.
    Returns first reason found or None if clean.
    """
    for check in [
        check_length,
        check_repeated_chars,
        check_prompt_injection,
        check_offensive,
        check_gibberish,
    ]:
        reason = check(message)
        if reason:
            return reason
    return None