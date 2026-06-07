"""
routers/recommendations.py

Recommendation endpoints called by Drupal on user login and product interaction.
"""

from fastapi import APIRouter, HTTPException
from models.schemas import LoginRecommendRequest, RecommendResponse
from services.recommendations import get_recommendations_for_user
from db.products import (
    log_user_interaction,
    invalidate_recommendations_cache,
    get_user_interactions,
    get_similar_products,
)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# Interaction weights — higher score = stronger preference signal
_INTERACTION_SCORES = {
    "view":        1.0,
    "click":       2.0,
    "add_to_cart": 3.0,
    "purchase":    5.0,
}


@router.post("/login", response_model=RecommendResponse)
async def recommendations_on_login(data: LoginRecommendRequest):
    """
    Called by Drupal hook_user_login.

    Returns personalized product recommendations for the user.
    New users get popular products; returning users get content-based
    recommendations blended with popular products.
    """
    try:
        result = await get_recommendations_for_user(data.user_id, limit=data.limit)
        return RecommendResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@router.get("/user/{user_id}", response_model=RecommendResponse)
async def get_user_recommendations(user_id: int, limit: int = 10):
    """
    Fetch fresh or cached recommendations for a user.
    Used by Drupal's AJAX endpoint to load recommendations on-demand.
    """
    try:
        result = await get_recommendations_for_user(user_id, limit=limit)
        return RecommendResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@router.post("/interact")
async def record_interaction(user_id: int, product_id: int,
                              interaction_type: str = "view"):
    """
    Record a user-product interaction to improve future recommendations.

    Interaction types: view, click, add_to_cart, purchase
    Stronger interactions (purchase > add_to_cart > click > view) have higher
    weight in the preference vector calculation.

    Also clears the recommendation cache for this user so the next
    /recommendations/login call returns fresh results.
    """
    if interaction_type not in _INTERACTION_SCORES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid interaction_type. Use one of: {list(_INTERACTION_SCORES)}"
        )

    score = _INTERACTION_SCORES[interaction_type]
    await log_user_interaction(user_id, product_id, interaction_type, score)
    await invalidate_recommendations_cache(user_id)

    return {"status": "recorded", "user_id": user_id,
            "product_id": product_id, "interaction_type": interaction_type}


@router.get("/explain/{user_id}")
async def explain_recommendations(user_id: int, limit: int = 10):
    """
    Show how recommendations are built for a user — exposes the vector math.

    Returns:
      - interactions_used: products the user touched (the "input signal")
      - ranked_products:   all candidates sorted by similarity_score DESC
                           so you can see P4/P2 score high, P3 scores low

    Example flow after seeding demo products and recording a view on iPhone (P1):
        interactions_used → [iPhone 15]
        ranked_products   → AirPods (0.96), Samsung (0.93), Nike (0.41)
    """
    try:
        interactions = await get_user_interactions(user_id, limit=5)
        if not interactions:
            return {
                "user_id": user_id,
                "message": "No interactions yet — record one via POST /recommendations/interact",
                "interactions_used": [],
                "ranked_products": [],
            }

        # Fetch similar products without excluding already-interacted ones
        # so the full ranking (including the seed product itself) is visible.
        ranked = await get_similar_products(user_id, limit=limit,
                                            exclude_interacted=False)

        return {
            "user_id": user_id,
            "interactions_used": [
                {"title": i["title"], "type": i["interaction_type"]}
                for i in interactions
            ],
            "ranked_products": [
                {
                    "product_id":       p["product_id"],
                    "title":            p["title"],
                    "category":         p["category"],
                    "similarity_score": round(float(p["similarity_score"]), 4),
                }
                for p in ranked
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explain failed: {str(e)}")
