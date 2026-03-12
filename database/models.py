from sqlalchemy import Column, Integer, String, DateTime, Date, ForeignKey, text, BigInteger
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .connection import Base


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    title = Column(String, nullable=False, server_default="New Chat")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(BigInteger, primary_key=True, server_default=text("GENERATED ALWAYS AS IDENTITY"))
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String, nullable=False)
    content = Column(String, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    attachments = Column(JSONB, server_default=text("'[]'::jsonb"))
    thinking = Column(JSONB, server_default=text("'[]'::jsonb"))
    message_metadata = Column("message_metadata", JSONB, server_default=text("'{}'::jsonb"))

    session = relationship("ChatSession", back_populates="messages")


class Patient(Base):
    __tablename__ = "patients"

    id             = Column(UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    patient_id     = Column(String, unique=True, nullable=False, index=True)
    full_name      = Column(String, nullable=False)
    date_of_birth  = Column(Date)
    age            = Column(Integer)
    sex            = Column(String)
    blood_type     = Column(String)
    race           = Column(String)
    ethnicity      = Column(String)
    marital_status = Column(String)
    address        = Column(JSONB, server_default=text("'{}'::jsonb"))
    diagnoses      = Column(JSONB, server_default=text("'[]'::jsonb"))
    medications    = Column(JSONB, server_default=text("'[]'::jsonb"))
    allergies      = Column(JSONB, server_default=text("'[]'::jsonb"))
    vitals_history = Column(JSONB, server_default=text("'[]'::jsonb"))
    last_visit     = Column(Date)
    created_at     = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at     = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
