import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from config import settings

async def migrate():
    print(f"Connecting to database: {settings.DATABASE_URL}")
    engine = create_async_engine(settings.DATABASE_URL)
    
    async with engine.begin() as conn:
        try:
            print("Checking if 'thinking' column exists in 'chat_messages' table...")
            # Check if column exists
            result = await conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='chat_messages' AND column_name='thinking';"))
            if result.scalar():
                print("Column 'thinking' already exists.")
            else:
                print("Adding 'thinking' column...")
                await conn.execute(text("ALTER TABLE chat_messages ADD COLUMN thinking JSONB DEFAULT '[]'::jsonb;"))
                print("Column 'thinking' added successfully.")
        except Exception as e:
            print(f"Migration failed: {e}")
            
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(migrate())
