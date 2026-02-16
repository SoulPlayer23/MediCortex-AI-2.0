import asyncio
import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/medicortex")

async def init_db():
    try:
        # asyncpg expects separate params or a DSN
        # sqlalchemy uses postgresql+asyncpg, but asyncpg expects postgresql
        clean_url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
        print(f"🔌 Connecting to {clean_url}...")
        conn = await asyncpg.connect(clean_url)
        print("✅ Connected to Database.")

        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        print(f"📖 Reading schema from {schema_path}...")
        
        with open(schema_path, "r") as f:
            schema_sql = f.read()

        print("🚀 Applying Schema...")
        await conn.execute(schema_sql)
        print("✅ Schema applied successfully.")
        
        await conn.close()
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(init_db())
