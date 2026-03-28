from fastapi import APIRouter
from models.schemas import GovernanceRequest
from services.governance import run_governance

router = APIRouter()

@router.post("/ai/governance")
def governance(data: GovernanceRequest):
    content = f"{data.title} {data.body}"
    return run_governance(content)