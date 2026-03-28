from pydantic import BaseModel, field_validator
from typing import List, Optional


class NodeData(BaseModel):
    """Request model for storing a Drupal node."""
    node_id: int
    title:   str
    content: str


class ChatRequest(BaseModel):
    """Request model for chatbot endpoint."""
    question: str

    @field_validator("question")
    def must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


class SearchQuery(BaseModel):
    """Request model for semantic search."""
    query: str


class SeoRequest(BaseModel):
    """Request model for SEO generation."""
    node_id: int
    title:   str
    content: str


class SeoResponse(BaseModel):
    """Response model for SEO generation."""
    node_id:    int
    meta_title: str
    meta_desc:  str
    keywords:   str
    cached:     bool = False


class ChatResponse(BaseModel):
    """Response model for chatbot endpoint."""
    question: str
    answer:   str
    sources:  List[dict]


class StoreResponse(BaseModel):
    """Response model for store endpoint."""
    status:  str
    node_id: int

class TextRequest(BaseModel):
    text: str

class GovernanceRequest(BaseModel):
    title: str
    body: str