"""
services/recommendations.py

Product recommendation engine.

Two strategies:
  popular      — new user with no history → return top products by interaction score
  personalized — returning user           → blend vector-similar products + popular fill

Results are cached per user for 30 minutes.
"""

from decimal import Decimal
from datetime import datetime
from typing import List

from db.products import (
    get_popular_products,
    get_similar_products,
    get_user_interactions,
    get_recommendations_cache,
    set_recommendations_cache,
)


def _serialize(products: List[dict]) -> List[dict]:
    """
    Convert DB row values to JSON-safe types.
    asyncpg returns Decimal for DECIMAL columns and datetime for TIMESTAMP.
    """
    result = []
    for p in products:
        item = {}
        for k, v in p.items():
            if isinstance(v, Decimal):
                item[k] = float(v)
            elif isinstance(v, datetime):
                item[k] = v.isoformat()
            else:
                item[k] = v
        result.append(item)
    return result


async def get_recommendations_for_user(user_id: int, limit: int = 10) -> dict:
    """
    Generate product recommendations for a Drupal user.

    Called on login via POST /recommendations/login.

    Flow:
        1. Check recommendation_cache → return immediately if fresh
        2. Check user_interactions → decide strategy
        3a. No history  → popular products (cold-start)
        3b. Has history → 70% content-based similar + 30% popular fill
        4. Cache result for 30 minutes
        5. Return

    Why 70/30 split:
        Pure similarity can create a filter bubble (only shows what they've
        already seen variations of). 30% popular ensures diversity and
        surfaces trending products the user might not have discovered.
    """
    # ── Cache hit ─────────────────────────────────────────────
    cached = await get_recommendations_cache(user_id)
    if cached:
        return {
            "user_id":  user_id,
            "products": cached,
            "strategy": "cached",
            "count":    len(cached),
        }

    # ── Check if user has any history ─────────────────────────
    interactions = await get_user_interactions(user_id, limit=5)

    if not interactions:
        # ── Cold start: new/inactive user ─────────────────────
        products  = await get_popular_products(limit=limit)
        strategy  = "popular"
    else:
        # ── Personalized: returning user ──────────────────────
        similar_limit  = max(1, int(limit * 0.7))
        popular_limit  = limit - similar_limit

        similar  = await get_similar_products(user_id, limit=similar_limit)
        popular  = await get_popular_products(limit=limit)   # fetch more to fill

        # Fill remaining slots with popular products not already in similar
        similar_ids   = {p["product_id"] for p in similar}
        extra_popular = [p for p in popular if p["product_id"] not in similar_ids]
        products      = similar + extra_popular[:popular_limit]
        strategy      = "personalized"

    # ── Serialize and cache ────────────────────────────────────
    serialized = _serialize(products)
    await set_recommendations_cache(user_id, serialized)

    return {
        "user_id":  user_id,
        "products": serialized,
        "strategy": strategy,
        "count":    len(serialized),
    }
