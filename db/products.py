"""
db/products.py

Database operations for ecommerce products, user interactions, and recommendations.
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from db.connection import get_pool
from services.embeddings import get_embedder


# ══════════════════════════════════════════════════════════════
# PRODUCTS — Store and retrieve product catalog
# ══════════════════════════════════════════════════════════════

async def store_product(product_id: int, title: str, description: str,
                        price: float, category: str, sku: str):
    """
    Store or update a product with its embedding.

    Embedding is built from: title + description + category
    so similarity search captures what the product IS, not just keywords.
    """
    pool = get_pool()

    text = f"{title}. {description}. Category: {category}".strip(". ")
    loop = asyncio.get_running_loop()
    embedding = await loop.run_in_executor(None, get_embedder().embed_query, text)

    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO products (product_id, title, description, price, category, sku, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7::vector)
            ON CONFLICT (product_id)
            DO UPDATE SET
                title       = EXCLUDED.title,
                description = EXCLUDED.description,
                price       = EXCLUDED.price,
                category    = EXCLUDED.category,
                sku         = EXCLUDED.sku,
                embedding   = EXCLUDED.embedding,
                updated_at  = CURRENT_TIMESTAMP;
        """, product_id, title, description, float(price), category, sku, str(embedding))


async def get_product_by_id(product_id: int) -> Optional[dict]:
    """Fetch a single product by its Drupal product ID."""
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT product_id, title, description, price, category, sku, created_at
            FROM products
            WHERE product_id = $1;
        """, product_id)
        return dict(row) if row else None


# ══════════════════════════════════════════════════════════════
# USER INTERACTIONS — Track what users view/click/buy
# ══════════════════════════════════════════════════════════════

async def log_user_interaction(user_id: int, product_id: int,
                                interaction_type: str, score: float = 1.0):
    """
    Record a user interaction with a product.

    Interaction types and weights:
        view         → 1.0  (saw the product)
        click        → 2.0  (clicked for details)
        add_to_cart  → 3.0  (showed buying intent)
        purchase     → 5.0  (strongest signal)
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO user_interactions (user_id, product_id, interaction_type, score)
            VALUES ($1, $2, $3, $4);
        """, user_id, product_id, interaction_type, float(score))


async def get_user_interactions(user_id: int, limit: int = 20) -> List[dict]:
    """Get the most recent product interactions for a user."""
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT ui.product_id, p.title, p.category,
                   ui.interaction_type, ui.score, ui.created_at
            FROM user_interactions ui
            JOIN products p ON p.product_id = ui.product_id
            WHERE ui.user_id = $1
            ORDER BY ui.created_at DESC
            LIMIT $2;
        """, user_id, limit)
        return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════
# POPULAR PRODUCTS — Fallback for new users
# ══════════════════════════════════════════════════════════════

async def get_popular_products(limit: int = 10) -> List[dict]:
    """
    Return most popular products by total interaction score.
    Used as fallback for new users with no history.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT p.product_id, p.title, p.description,
                   p.price, p.category, p.sku,
                   COALESCE(SUM(ui.score), 0) AS popularity_score
            FROM products p
            LEFT JOIN user_interactions ui ON ui.product_id = p.product_id
            GROUP BY p.product_id, p.title, p.description,
                     p.price, p.category, p.sku, p.created_at
            ORDER BY popularity_score DESC, p.created_at DESC
            LIMIT $1;
        """, limit)
        return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════
# CONTENT-BASED RECOMMENDATIONS — Vector similarity
# ══════════════════════════════════════════════════════════════

async def get_similar_products(user_id: int, limit: int = 10,
                                exclude_interacted: bool = True) -> List[dict]:
    """
    Content-based recommendations using pgvector.

    Strategy:
    1. Fetch embeddings of products the user interacted with (recent 10, weighted by score)
    2. Compute a weighted average → user preference vector
    3. Find products most similar to this vector via cosine similarity
    4. Exclude products the user already interacted with

    Why weighted average:
        A purchase (score=5) should pull the preference vector more than a view (score=1).
        This makes the recommendation track actual intent, not just browsing history.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT p.embedding, ui.score
            FROM user_interactions ui
            JOIN products p ON p.product_id = ui.product_id
            WHERE ui.user_id = $1
            ORDER BY ui.score DESC, ui.created_at DESC
            LIMIT 10;
        """, user_id)

        if not rows:
            return []

        # Parse vector strings and build weighted average
        embeddings = []
        weights = []
        for row in rows:
            emb_str = str(row["embedding"])
            emb = [float(x) for x in emb_str.strip("[]").split(",")]
            embeddings.append(emb)
            weights.append(float(row["score"]))

        weights_arr = np.array(weights, dtype=float)
        weights_arr /= weights_arr.sum()
        pref_vector = np.average(np.array(embeddings, dtype=float),
                                  axis=0, weights=weights_arr)
        pref_str = "[" + ",".join(f"{v:.6f}" for v in pref_vector.tolist()) + "]"

        if exclude_interacted:
            similar_rows = await conn.fetch("""
                SELECT p.product_id, p.title, p.description,
                       p.price, p.category, p.sku,
                       1 - (p.embedding <=> $1::vector) AS similarity_score
                FROM products p
                WHERE p.product_id NOT IN (
                    SELECT DISTINCT product_id
                    FROM user_interactions
                    WHERE user_id = $2
                )
                ORDER BY p.embedding <=> $1::vector
                LIMIT $3;
            """, pref_str, user_id, limit)
        else:
            similar_rows = await conn.fetch("""
                SELECT p.product_id, p.title, p.description,
                       p.price, p.category, p.sku,
                       1 - (p.embedding <=> $1::vector) AS similarity_score
                FROM products p
                ORDER BY p.embedding <=> $1::vector
                LIMIT $2;
            """, pref_str, limit)

        return [dict(r) for r in similar_rows]


# ══════════════════════════════════════════════════════════════
# RECOMMENDATION CACHE — Avoid hitting Python on every page
# ══════════════════════════════════════════════════════════════

async def get_recommendations_cache(user_id: int) -> Optional[List[dict]]:
    """Return cached recommendations if not expired (30-minute TTL)."""
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT recommendations
            FROM recommendation_cache
            WHERE user_id = $1 AND expires_at > NOW();
        """, user_id)
        if row:
            return json.loads(row["recommendations"])
    return None


async def set_recommendations_cache(user_id: int, recommendations: List[dict]):
    """Cache recommendations for 30 minutes."""
    pool = get_pool()
    expires_at = (datetime.now(timezone.utc).replace(tzinfo=None)
                  + timedelta(minutes=30))
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO recommendation_cache (user_id, recommendations, expires_at)
            VALUES ($1, $2, $3)
            ON CONFLICT (user_id)
            DO UPDATE SET
                recommendations = EXCLUDED.recommendations,
                expires_at      = EXCLUDED.expires_at,
                created_at      = CURRENT_TIMESTAMP;
        """, user_id, json.dumps(recommendations, default=str), expires_at)


async def invalidate_recommendations_cache(user_id: int):
    """Clear cached recommendations for a user (call after new interaction)."""
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM recommendation_cache WHERE user_id = $1;",
            user_id
        )
