import os
import aioboto3
import logging
from botocore.exceptions import ClientError
from typing import Optional

logger = logging.getLogger("MinioService")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET_NAME = "medicortex-uploads"

class MinioService:
    def __init__(self):
        self.session = aioboto3.Session()
        logger.info(f"MinIO Config: Endpoint={MINIO_ENDPOINT}, Key={MINIO_ACCESS_KEY}...")

    async def ensure_bucket_exists(self):
        from botocore.client import Config
        async with self.session.client("s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            region_name="us-east-1",
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        ) as s3:
            try:
                await s3.head_bucket(Bucket=BUCKET_NAME)
                logger.info(f"✅ MinIO Bucket '{BUCKET_NAME}' exists.")
            except ClientError:
                logger.info(f"⚠️ Bucket '{BUCKET_NAME}' not found. Creating...")
                await s3.create_bucket(Bucket=BUCKET_NAME)
                logger.info(f"✅ Bucket '{BUCKET_NAME}' created.")

    async def upload_file(self, file_data: bytes, filename: str, content_type: str) -> Optional[str]:
        from botocore.client import Config
        async with self.session.client("s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            region_name="us-east-1",
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        ) as s3:
            try:
                await s3.put_object(
                    Bucket=BUCKET_NAME,
                    Key=filename,
                    Body=file_data,
                    ContentType=content_type
                )
                # Generate Presigned URL (valid for 7 days)
                url = await s3.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': BUCKET_NAME, 'Key': filename},
                    ExpiresIn=604800
                )
                return url
            except Exception as e:
                logger.error(f"❌ MinIO Upload Failed: {e}")
                return None

minio_service = MinioService()
