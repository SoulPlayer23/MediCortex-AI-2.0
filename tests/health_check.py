import asyncio
import os
import sys
import httpx
import logging
from datetime import datetime
from sqlalchemy import text
from typing import Optional

# Ensure project root is in path
sys.path.append(os.getcwd())

import dotenv
dotenv.load_dotenv()

# Services
# Note: These imports might rely on other modules being available
try:
    from database.connection import get_db, engine
    from services.minio_service import minio_service
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker
except ImportError as e:
    print(f"❌ Failed to import dependencies: {e}")
    sys.exit(1)

# Configuration
ORCHESTRATOR_URL = "http://localhost:8001"
TIMEOUT = 5.0

class HealthCheck:
    def __init__(self):
        self.results = []
        # Setup logging for cleanliness
        logging.basicConfig(level=logging.ERROR)

    def log(self, message, status="INFO"):
        icon_map = {
            "INFO": "ℹ️ ",
            "PASS": "✅",
            "FAIL": "❌",
            "WARN": "⚠️ "
        }
        icon = icon_map.get(status, "")
        print(f" {icon} {message}")
        
    async def check_database(self) -> bool:
        print("\n[Database]")
        try:
            # Create a localized session for the check
            async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            async with async_session_maker() as session:
                await session.execute(text("SELECT 1"))
            self.log("PostgreSQL Connection Successful", "PASS")
            return True
        except Exception as e:
            self.log(f"Database Connection Failed: {e}", "FAIL")
            return False

    async def check_minio(self) -> bool:
        print("\n[Storage]")
        try:
            # Check connection by ensuring bucket exists
            self.log("Checking MinIO Bucket...", "INFO")
            await minio_service.ensure_bucket_exists()
            
            # Test Upload
            self.log("Testing Upload...", "INFO")
            test_content = b"Health Check Probe"
            filename = f"health_check_{int(datetime.now().timestamp())}.txt"
            url = await minio_service.upload_file(test_content, filename, "text/plain")
            
            if url:
                 self.log(f"Upload Successful: {url[:50]}...", "PASS")
                 return True
            else:
                 self.log("Upload Failed (No URL returned)", "FAIL")
                 return False
                 
        except Exception as e:
            self.log(f"MinIO Check Failed: {e}", "FAIL")
            return False

    async def check_orchestrator(self) -> bool:
        print("\n[Orchestrator API]")
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.get(f"{ORCHESTRATOR_URL}/health")
                if resp.status_code == 200:
                    data = resp.json()
                    agents = data.get("agents", [])
                    agent_list = ", ".join(agents) if agents else "None"
                    self.log(f"Status 200 OK | Active Agents: {len(agents)}", "PASS")
                    return True
                else:
                    self.log(f"Returned Status {resp.status_code}", "FAIL")
                    return False
        except httpx.ConnectError:
             self.log(f"Connection Refused at {ORCHESTRATOR_URL}", "FAIL")
             self.log("Is the orchestrator.py service running?", "WARN")
             return False
        except Exception as e:
             self.log(f"Check Failed: {e}", "FAIL")
             return False

    async def run(self):
        print("🏥 MediCortex System Health Check")
        print("="*40)
        
        db_ok = await self.check_database()
        minio_ok = await self.check_minio()
        orch_ok = await self.check_orchestrator()
        
        print("\n" + "="*40)
        checks = [
            ("Database", db_ok), 
            ("Storage", minio_ok), 
            ("Orchestrator", orch_ok)
        ]
        
        all_passed = all(status for _, status in checks)
        
        print(f"Summary: {'✅ HEALTHY' if all_passed else '❌ UNHEALTHY'}")
        
        if not all_passed:
            sys.exit(1)

if __name__ == "__main__":
    checker = HealthCheck()
    try:
        asyncio.run(checker.run())
    except KeyboardInterrupt:
        print("\n⚠️ Check interrupted.")
