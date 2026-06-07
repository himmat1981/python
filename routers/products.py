"""
routers/products.py

Product catalog endpoints.
Called by Drupal's ProductSyncWorker queue worker when a product node is
created or updated.
"""

from fastapi import APIRouter, HTTPException
from models.schemas import ProductData, ProductStoreResponse
from db.products import store_product, get_popular_products, get_product_by_id

router = APIRouter(prefix="/products", tags=["products"])


@router.post("/store", response_model=ProductStoreResponse)
async def handle_store_product(data: ProductData):
    """
    Store or update a product from Drupal.

    Called by Drupal's ProductSyncWorker after a product node is saved.
    Generates an embedding from title + description + category and stores
    it in the products table for similarity-based recommendations.
    """
    try:
        await store_product(
            product_id  = data.product_id,
            title       = data.title,
            description = data.description,
            price       = data.price,
            category    = data.category,
            sku         = data.sku,
        )
        return ProductStoreResponse(status="stored", product_id=data.product_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Product storage failed: {str(e)}")


@router.get("/list")
async def list_products(limit: int = 20):
    """List products ordered by popularity. Useful for Drupal admin checks."""
    try:
        products = await get_popular_products(limit=limit)
        return {"products": products, "count": len(products)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{product_id}")
async def get_product(product_id: int):
    """Get a single product by its Drupal product ID."""
    try:
        product = await get_product_by_id(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        return product
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
