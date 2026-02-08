import asyncio
import httpx
import os

BASE_URL = "http://localhost:8001"

async def test_flow():
    async with httpx.AsyncClient(timeout=60.0) as client:
        print("1. Checking Health...")
        try:
            resp = await client.get(f"{BASE_URL}/health")
            print(f"Health: {resp.status_code} - {resp.json()}")
        except Exception as e:
            print(f"❌ Health Check Failed: {e}")
            return

        print("\n2. Creating New Chat Session (via message)...")
        # Sending a message without session_id should create one
        msg_payload = {"message": "What are the key symptoms of Influenza?"}
        resp = await client.post(f"{BASE_URL}/chat", json=msg_payload)
        if resp.status_code != 200:
            print(f"❌ Chat Failed: {resp.text}")
            return
        
        data = resp.json()
        session_id = data.get("session_id")
        print(f"✅ Chat Response Received. Session ID: {session_id}")
        print(f"Response: {data.get('response')[:50]}...")

        print(f"\n3. Fetching Chat History for Session {session_id}...")
        resp = await client.get(f"{BASE_URL}/chats/{session_id}")
        history = resp.json()
        print(f"✅ History Fetched. Count: {len(history)}")
        for msg in history:
            print(f" - [{msg['role']}]: {msg['content'][:30]}...")

        print("\n4. Fetching All Sessions...")
        resp = await client.get(f"{BASE_URL}/chats")
        sessions = resp.json()
        print(f"✅ Sessions Fetched. Count: {len(sessions)}")
        found = any(s['id'] == session_id for s in sessions)
        print(f"Current session found in list: {found}")

        print("\n5. Testing File Upload...")
        # Create a dummy file
        files = {'file': ('test_image.txt', b'This is a test file content', 'text/plain')}
        resp = await client.post(f"{BASE_URL}/upload", files=files)
        if resp.status_code == 200:
            file_data = resp.json()
            print(f"✅ Upload Successful: {file_data}")
        else:
            print(f"❌ Upload Failed: {resp.text}")

if __name__ == "__main__":
    asyncio.run(test_flow())
