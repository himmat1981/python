"""
routers/nodes.py

Store Drupal nodes with smart chunking.
Each node is split into overlapping chunks before embedding.
"""

from fastapi import APIRouter, HTTPException
from models.schemas import NodeData, StoreResponse
from services.chunker import chunk_node
from db.vectors import store_node_chunks

router = APIRouter(prefix="/nodes", tags=["nodes"])


@router.post("/store", response_model=StoreResponse)
async def handle_store_node(data: NodeData):
    """
    Store a Drupal node with smart chunking.

    Flow:
    1. Split content into 300-word chunks with 50-word overlap
    2. Generate embedding for each chunk
    3. Store all chunks in node_chunks table

    Called by Drupal hook_node_insert / hook_node_update.
    """
    try:
        chunks = chunk_node(data.node_id, data.title, data.content)
        await store_node_chunks(chunks)

        return StoreResponse(
            status  = f"stored {len(chunks)} chunks successfully",
            node_id = data.node_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage failed: {str(e)}")