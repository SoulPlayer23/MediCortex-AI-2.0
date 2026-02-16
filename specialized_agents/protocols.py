import uuid
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

# --- A2A Protocol Models ---

class AgentCard(BaseModel):
    """
    Metadata manifest that describes an agent's capabilities.
    Each agent MUST serve this card.
    """
    name: str = Field(..., description="Unique name of the agent (e.g., 'pubmed-researcher')")
    description: str = Field(..., description="Human-readable description of what the agent does")
    input_schema: Dict[str, Any] = Field(..., description="JSON Schema of the input payload accepted")
    output_schema: Dict[str, Any] = Field(..., description="JSON Schema of the output payload produced")
    version: str = "1.0.0"
    capabilities: List[str] = Field(default_factory=list, description="List of capability tags")

class Envelope(BaseModel):
    """
    Standard Envelope for A2A communication.
    """
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    idempotency_key: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = "orchestrator"
    receiver_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload: Dict[str, Any] = Field(..., description="The actual data being passed")

class AgentResponse(BaseModel):
    """
    Standard Response from an Agent to the Sender.
    """
    envelope_id: str # Reference to the request envelope ID (idempotency_key)
    output: Any = Field(..., description="The result of the agent's work")
    error: Optional[str] = None
    usage: Dict[str, int] = Field(default_factory=dict, description="Token usage or cost metrics")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
