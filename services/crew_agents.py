from services.llm import chat
import json
import re

def parse_json_response(text: str):
    try:
        # Remove ```json ``` wrapper
        cleaned = re.sub(r"```json|```", "", text).strip()
        return json.loads(cleaned)
    except Exception:
        return {"raw": text}

from services.llm import chat
import json

def quality_agent(content: str):
    response = chat([
        {
            "role": "system",
            "content": "You are a content quality analyst. ONLY return valid JSON. No markdown, no explanation."
        },
        {
            "role": "user",
            "content": f"""
Return ONLY JSON in this format:
{{ "score": number, "issues": [] }}

Content:
{content}
"""
        }
    ])

    try:
        return json.loads(response)
    except:
        return {"raw": response}

def fact_checker_agent(content: str, context: str):
    response = chat([
        {
            "role": "system",
            "content": "You are a fact checker. ONLY return valid JSON."
        },
        {
            "role": "user",
            "content": f"""
Return ONLY JSON:
{{ "factual_accuracy": true/false, "incorrect_points": [] }}

Context:
{context}

Content:
{content}
"""
        }
    ])

    try:
        return json.loads(response)
    except:
        return {"raw": response}


import json
from services.llm import chat

def rewrite_agent(content: str):
    response = chat([
        {
            "role": "system",
            "content": "You are a professional content writer. ONLY return JSON. No explanation."
        },
        {
            "role": "user",
            "content": f"""
Rewrite the content to improve clarity, grammar, and engagement.

Return ONLY JSON:
{{ "improved_content": "..." }}

Content:
{content}
"""
        }
    ])

    try:
        return json.loads(response)
    except:
        return {"improved_content": response}


def compliance_agent(content: str):
    response = chat([
        {
            "role": "system",
            "content": "You are a compliance expert. ONLY return valid JSON."
        },
        {
            "role": "user",
            "content": f"""
Return ONLY JSON:
{{ "compliant": "yes/no", "issues": [] }}

Content:
{content}
"""
        }
    ])

    try:
        return json.loads(response)
    except:
        return {"raw": response}