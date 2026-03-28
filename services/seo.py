import json
import re
from typing import Optional
from services.llm import chat


def generate_seo_tags(title: str, content: str) -> dict:
    """
    Generate SEO meta title, meta description and keywords
    for a Drupal node using Groq LLM.

    Returns a dict with keys: meta_title, meta_desc, keywords
    """

    # Truncate content to avoid exceeding token limits
    # 1500 chars is enough context for SEO generation
    truncated_content = content[:1500] if len(content) > 1500 else content

    messages = [
        {
            "role": "system",
            "content": (
                "You are an SEO expert. When given article content, "
                "you generate optimized SEO metadata. "
                "Always respond with valid JSON only — no explanation, "
                "no markdown, no code blocks. Just raw JSON."
            )
        },
        {
            "role": "user",
            "content": f"""Generate SEO metadata for this article:

Title: {title}

Content: {truncated_content}

Respond with this exact JSON structure:
{{
    "meta_title": "SEO optimized title under 60 characters",
    "meta_desc": "Compelling meta description between 150-160 characters",
    "keywords": "keyword1, keyword2, keyword3, keyword4, keyword5"
}}"""
        }
    ]

    # Lower temperature for more consistent/factual output
    raw = chat(messages, max_tokens=300, temperature=0.3)

    return _parse_seo_response(raw, title)


def _parse_seo_response(raw: str, fallback_title: str) -> dict:
    """
    Safely parse the LLM JSON response.
    Falls back to safe defaults if parsing fails.
    """
    try:
        # Strip markdown code blocks if LLM added them despite instructions
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(clean)

        return {
            "meta_title": data.get("meta_title", fallback_title)[:60],
            "meta_desc":  data.get("meta_desc", "")[:160],
            "keywords":   data.get("keywords", ""),
        }
    except (json.JSONDecodeError, Exception):
        # If LLM returns invalid JSON, return safe fallbacks
        return {
            "meta_title": fallback_title[:60],
            "meta_desc":  "",
            "keywords":   "",
        }