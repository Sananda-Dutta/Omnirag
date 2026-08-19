import uuid

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    knowledge_base_id: uuid.UUID | None = Field(
        default=None, description="Restrict retrieval to one knowledge base. Omit to search all of the user's."
    )


class CitationItem(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_filename: str
    chunk_index: int
    score: float
    text_snippet: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationItem]
    model_used: str
    context_found: bool
