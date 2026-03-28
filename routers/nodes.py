from fastapi import APIRouter, HTTPException
from models.schemas import NodeData, StoreResponse
from services.embeddings import encode
from db.vectors import store_node_vector

router = APIRouter(prefix="/nodes", tags=["nodes"])


@router.post("/store", response_model=StoreResponse)
async def store_node(data: NodeData):
    """
    Store a Drupal node with its vector embedding.
    Called by Drupal hook_node_insert / hook_node_update.
    """
    try:
        embedding = encode(data.content)
        store_node_vector(data.node_id, data.title, data.content, embedding)
        return StoreResponse(status="stored successfully", node_id=data.node_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage failed: {str(e)}")