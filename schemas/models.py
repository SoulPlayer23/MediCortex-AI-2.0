from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Any
from datetime import datetime
from uuid import UUID

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[UUID] = None

class MessageAttachment(BaseModel):
    filename: str
    url: str
    content_type: str
    
class MessageResponse(BaseModel):
    id: int
    session_id: UUID
    role: str
    content: str
    timestamp: datetime
    attachments: List[Any] = Field(default_factory=list) # JSONB content
    thinking: List[str] = Field(default_factory=list) # JSONB content
    
    model_config = ConfigDict(from_attributes=True)

class SessionResponse(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class ChatResponse(BaseModel):
    response: str
    session_id: UUID
    thinking: List[str] = Field(default_factory=list)

class UploadResponse(BaseModel):
    url: str
    filename: str

class HealthResponse(BaseModel):
    status: str
    agents: List[str]
