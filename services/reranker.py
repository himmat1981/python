"""
services/reranker.py

Rerank search results after initial vector search.

Why reranking matters:
  Vector search finds "similar" documents by cosine distance.
  But similar != most relevant for the specific question.

  Example:
    Query: "how to reset drupal password"
    Vector search top-3:
      1. "User account management in Drupal"  (similarity: 0.82)
      2. "Password reset via drush"           (similarity: 0.79)  ← actually best answer
      3. "Drupal security best practices"     (similarity: 0.76)

  After reranking by keyword overlap + similarity combined:
      1. "Password reset via drush"           (score: 0.91)  ← now #1 ✅
      2. "User account management in Drupal"  (score: 0.74)
      3. "Drupal security best practices"     (score: 0.61)

Strategy used here: lightweight scoring without extra model.
  Score = (0.6 × vector_similarity) + (0.4 × keyword_overlap)
  Fast, no extra API calls, no extra model needed.
"""

from typing import List
import re


def _keyword_overlap_score(query: str, content: str) -> float:
    """
    Calculate what fraction of query words appear in content.
    Simple but effective for exact term matching.
    """
    # Normalize both texts
    query_words   = set(re.findall(r'\w+', query.lower()))
    content_words = set(re.findall(r'\w+', content.lower()))

    # Remove common stop words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into", "what",
        "how", "why", "when", "where", "who", "which", "that", "this",
    }
    query_words -= stop_words

    if not query_words:
        return 0.0

    # Count how many query words appear in content
    matches = query_words.intersection(content_words)
    return len(matches) / len(query_words)


def rerank(
    query: str,
    results: List[dict],
    top_k: int = 3,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
) -> List[dict]:
    """
    Rerank search results using combined score.

    Args:
        query:          Original user question
        results:        List of dicts with 'content', 'similarity', etc.
        top_k:          How many results to return after reranking
        vector_weight:  Weight for vector similarity (intent-driven)
        keyword_weight: Weight for keyword overlap  (intent-driven)

    Returns:
        Top-k results sorted by combined relevance score
    """
    if not results:
        return []

    scored = []
    for result in results:
        content    = result.get("content", "")
        similarity = float(result.get("similarity", 0.0))
        keyword    = _keyword_overlap_score(query, content)

        # Combined score weighted by intent
        combined_score = (vector_weight * similarity) + (keyword_weight * keyword)

        scored.append({
            **result,
            "rerank_score":    round(combined_score, 4),
            "keyword_overlap": round(keyword, 4),
        })

    # Sort by combined score descending
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)

    return scored[:top_k]