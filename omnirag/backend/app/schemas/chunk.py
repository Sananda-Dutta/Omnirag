"""
Chunk response schema.

`embedding` is deliberately NOT included here. It's a 384-to-1536-element
float array — meaningless to a human reading the API response, and there's
no legitimate client use case for shipping it over the wire at this stage
(Phase 6 does similarity search server-side, inside the vector DB). Exposing
it would just bloat every response for no benefit.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ChunkRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    chunk_index: int
    text: str
    char_count: int
    embedding_model: str
    embedding_dimension: int
    created_at: datetime
