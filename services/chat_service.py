from sqlalchemy import select, desc, text
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

    async def get_sessions(self, db: AsyncSession) -> List[dict]:
        """Return sessions ordered by most-recently-updated, each with a one-line
        preview taken from the last message in that session."""
        stmt = text("""
            SELECT s.id, s.title, s.created_at, s.updated_at,
                   COALESCE(m.content, '') AS last_message
            FROM chat_sessions s
            LEFT JOIN LATERAL (
                SELECT content FROM chat_messages
                WHERE session_id = s.id
                ORDER BY timestamp DESC
                LIMIT 1
            ) m ON true
            ORDER BY s.updated_at DESC
        """)
        rows = (await db.execute(stmt)).fetchall()
        return [
            {
                "id": row.id,
                "title": row.title,
                "preview": row.last_message[:80] if row.last_message else "",
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            }
            for row in rows
        ]

    async def get_session(self, db: AsyncSession, session_id: str) -> Optional[ChatSession]:
        result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
        return result.scalar_one_or_none()

    async def add_message(self, db: AsyncSession, session_id: str, role: str, content: str, attachments: list = None, thinking: list = None, metadata: dict = None) -> ChatMessage:
        if attachments is None:
            attachments = []
        if thinking is None:
            thinking = []
        if metadata is None:
            metadata = {}
            
        new_message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            attachments=attachments,
            thinking=thinking,
            message_metadata=metadata
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
