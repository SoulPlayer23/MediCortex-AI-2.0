from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from database.models import ChatSession, ChatMessage
from database.connection import get_db
import uuid
from typing import List, Optional

class ChatService:
    
    async def create_session(self, db: AsyncSession, title: str = "New Chat") -> ChatSession:
        session_id = str(uuid.uuid4())
        new_session = ChatSession(id=session_id, title=title)
        db.add(new_session)
        await db.commit()
        await db.refresh(new_session)
        return new_session

    async def get_sessions(self, db: AsyncSession) -> List[ChatSession]:
        result = await db.execute(select(ChatSession).order_by(desc(ChatSession.updated_at)))
        return result.scalars().all()

    async def get_session(self, db: AsyncSession, session_id: str) -> Optional[ChatSession]:
        result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
        return result.scalar_one_or_none()

    async def add_message(self, db: AsyncSession, session_id: str, role: str, content: str, attachments: list = None, thinking: list = None) -> ChatMessage:
        if attachments is None:
            attachments = []
        if thinking is None:
            thinking = []
            
        new_message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            attachments=attachments,
            thinking=thinking
        )
        db.add(new_message)
        
        # Update session timestamp
        session = await self.get_session(db, session_id)
        if session:
            # Auto-generate title for first user message if title is 'New Chat'
            if session.title == "New Chat" and role == "user":
                session.title = content[:30] + "..." if len(content) > 30 else content
            
            from datetime import datetime
            session.updated_at = datetime.utcnow()
            
        await db.commit()
        await db.refresh(new_message)
        return new_message

    async def get_messages(self, db: AsyncSession, session_id: str) -> List[ChatMessage]:
        result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.timestamp)
        )
        return result.scalars().all()

chat_service = ChatService()
