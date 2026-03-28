from fastapi import APIRouter, HTTPException
from models.schemas import SeoRequest, SeoResponse
from services.seo import generate_seo_tags
from db.vectors import save_seo_cache, get_seo_cache

router = APIRouter(prefix="/seo", tags=["seo"])


@router.post("/generate", response_model=SeoResponse)
async def generate_seo(data: SeoRequest):
    """
    Generate SEO meta title, description and keywords
    for a Drupal node using Groq LLM.

    First checks cache — if tags already exist for this node,
    returns cached version without calling LLM again.

    Called by Drupal hook_node_insert / hook_node_update.
    """
    try:
        # Check cache first — avoid unnecessary LLM calls
        cached = get_seo_cache(data.node_id)
        if cached:
            return SeoResponse(
                node_id    = data.node_id,
                meta_title = cached["meta_title"],
                meta_desc  = cached["meta_desc"],
                keywords   = cached["keywords"],
                cached     = True,
            )

        # Generate fresh SEO tags via LLM
        seo = generate_seo_tags(data.title, data.content)

        # Save to cache for future requests
        save_seo_cache(
            data.node_id,
            seo["meta_title"],
            seo["meta_desc"],
            seo["keywords"],
        )

        return SeoResponse(
            node_id    = data.node_id,
            meta_title = seo["meta_title"],
            meta_desc  = seo["meta_desc"],
            keywords   = seo["keywords"],
            cached     = False,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SEO generation failed: {str(e)}")


@router.get("/cached/{node_id}", response_model=SeoResponse)
async def get_cached_seo(node_id: int):
    """
    Retrieve cached SEO tags for a node.
    Returns 404 if not yet generated.
    """
    cached = get_seo_cache(node_id)
    if not cached:
        raise HTTPException(
            status_code=404,
            detail=f"No SEO tags found for node {node_id}. Call POST /seo/generate first."
        )
    return SeoResponse(
        node_id    = node_id,
        meta_title = cached["meta_title"],
        meta_desc  = cached["meta_desc"],
        keywords   = cached["keywords"],
        cached     = True,
    )