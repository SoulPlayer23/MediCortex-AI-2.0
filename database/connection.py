from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager
from sqlalchemy.orm import declarative_base

from config import settings

# Create Async Engine
# OPS-1: Pool sizing for 10-user concurrency. Each /chat/stream request holds
# a connection for the full LangGraph pipeline (30–120s during MedGemma synthesis),
# so the default pool_size=5 is exhausted at the target scale.
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=settings.SQLALCHEMY_POOL_SIZE,
    max_overflow=settings.SQLALCHEMY_MAX_OVERFLOW,
    pool_timeout=settings.SQLALCHEMY_POOL_TIMEOUT,
    pool_pre_ping=True,  # detects stale connections after VPS/homeserver restarts
)
Base = declarative_base()

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
