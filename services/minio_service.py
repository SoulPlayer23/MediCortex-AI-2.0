import logging
from botocore.client import Config
from botocore.exceptions import ClientError
from typing import Optional

import aioboto3

from config import settings

logger = logging.getLogger("MinioService")


class MinioService:
    def __init__(self):
        self.session = aioboto3.Session()
        logger.info(f"MinIO Config: Endpoint={settings.MINIO_URL}, Key={settings.MINIO_ACCESS_KEY[:4]}...")

    def _client(self):
        return self.session.client(
            "s3",
            endpoint_url=settings.MINIO_URL,
            aws_access_key_id=settings.MINIO_ACCESS_KEY,
            aws_secret_access_key=settings.MINIO_SECRET_KEY,
            region_name="us-east-1",
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )

    async def ensure_bucket_exists(self):
        async with self._client() as s3:
            try:
                await s3.head_bucket(Bucket=settings.MINIO_BUCKET_NAME)
                logger.info(f"MinIO bucket '{settings.MINIO_BUCKET_NAME}' exists.")
            except ClientError:
                logger.info(f"Bucket '{settings.MINIO_BUCKET_NAME}' not found. Creating...")
                await s3.create_bucket(Bucket=settings.MINIO_BUCKET_NAME)
                logger.info(f"Bucket '{settings.MINIO_BUCKET_NAME}' created.")

    async def upload_file(self, file_data: bytes, filename: str, content_type: str) -> Optional[str]:
        async with self._client() as s3:
            try:
                await s3.put_object(
                    Bucket=settings.MINIO_BUCKET_NAME,
                    Key=filename,
                    Body=file_data,
                    ContentType=content_type,
                )
                url = await s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": settings.MINIO_BUCKET_NAME, "Key": filename},
                    ExpiresIn=604800,  # 7 days
                )
                return url
            except Exception as e:
                logger.error(f"MinIO upload failed: {e}")
                return None


minio_service = MinioService()
