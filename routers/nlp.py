from fastapi import APIRouter
from models.schemas import TextRequest
from services.nlp import summarize_text, moderate_text

router = APIRouter(prefix="/nlp", tags=["NLP"])


@router.post("/summarize")
def summarize(request: TextRequest):
    summary = summarize_text(request.text)
    return {"summary": summary}


@router.post("/moderate")
def moderate(request: TextRequest):
    result = moderate_text(request.text)
    return result