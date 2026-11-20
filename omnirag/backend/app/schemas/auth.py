"""
Request/response schemas for auth endpoints.

These are deliberately separate from `app.models.User` (the SQLAlchemy
table). Mixing ORM models directly into API responses is how you
accidentally serialize `hashed_password` to a client — a schema layer makes
"what the API exposes" an explicit, reviewable contract instead of
"whatever attributes happen to be on the DB object."
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    is_active: bool
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
