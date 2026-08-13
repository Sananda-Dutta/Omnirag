import uuid

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    knowledge_base_id: uuid.UUID | None = Field(
        default=None, description="Restrict search to one knowledge base. Omit to search all of the user's."
    )
    top_k: int | None = Field(default=None, ge=1, le=50)


class SearchResultItem(BaseModel):
    chunk_id: uuid.UUID
    score: float
    text: str
    chunk_index: int
    document_id: uuid.UUID
    document_filename: str
