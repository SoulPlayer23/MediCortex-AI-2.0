import logging
from urllib.parse import urlparse, urlunparse
from botocore.client import Config
from botocore.exceptions import ClientError
from typing import Optional

import aioboto3

from config import settings

logger = logging.getLogger("MinioService")


class MinioService:
    def __init__(self):
        self.session = aioboto3.Session()
        logger.info(
            f"MinIO Config: Endpoint={settings.MINIO_URL}, Public={settings.MINIO_PUBLIC_URL or '(none)'}, "
            f"Key={settings.MINIO_ACCESS_KEY[:4] if settings.MINIO_ACCESS_KEY else '(unset)'}..."
        )

    def _client(self):
        return self.session.client(
            "s3",
            endpoint_url=settings.MINIO_URL,
            aws_access_key_id=settings.MINIO_ACCESS_KEY,
            aws_secret_access_key=settings.MINIO_SECRET_KEY,
            region_name="us-east-1",
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )

    @staticmethod
    def _swap_host(url: str, public_base: Optional[str]) -> str:
        """SEC-2: rewrite the host of a presigned URL to the user-reachable base.

        Internal MINIO_URL may use host.docker.internal or a tailnet-only host
        the browser cannot resolve. When MINIO_PUBLIC_URL is set, swap the
        scheme+netloc but preserve the path + signed query string.
        """
        if not public_base:
            return url
        try:
            src = urlparse(url)
            pub = urlparse(public_base)
            return urlunparse((pub.scheme, pub.netloc, src.path, src.params, src.query, src.fragment))
        except Exception:
            return url

    async def ensure_bucket_exists(self):
        async with self._client() as s3:
            try:
                await s3.head_bucket(Bucket=settings.MINIO_BUCKET_NAME)
                logger.info(f"MinIO bucket '{settings.MINIO_BUCKET_NAME}' exists.")
            except ClientError:
                logger.info(f"Bucket '{settings.MINIO_BUCKET_NAME}' not found. Creating...")
                await s3.create_bucket(Bucket=settings.MINIO_BUCKET_NAME)
                logger.info(f"Bucket '{settings.MINIO_BUCKET_NAME}' created.")

    async def upload_file(
        self, file_data: bytes, filename: str, content_type: str
    ) -> Optional[str]:
        async with self._client() as s3:
            try:
                await s3.put_object(
                    Bucket=settings.MINIO_BUCKET_NAME,
                    Key=filename,
                    Body=file_data,
                    ContentType=content_type,
                )
                # SEC-2: 1-hour TTL by default (was 7 days). Configurable via
                # MINIO_PRESIGN_TTL_SECONDS to cope with slow downstream paths
                # (e.g. RunPod cold-start MedGemma fetching the document).
                url = await s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": settings.MINIO_BUCKET_NAME, "Key": filename},
                    ExpiresIn=settings.MINIO_PRESIGN_TTL_SECONDS,
                )
                return self._swap_host(url, settings.MINIO_PUBLIC_URL)
            except Exception as e:
                logger.error(f"MinIO upload failed: {e}")
                return None

    async def generate_download_url(self, filename: str) -> Optional[str]:
        """Issue a fresh short-lived presigned URL for an existing object.

        Used by the UI to refresh download links without re-uploading.
        """
        async with self._client() as s3:
            try:
                url = await s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": settings.MINIO_BUCKET_NAME, "Key": filename},
                    ExpiresIn=settings.MINIO_PRESIGN_TTL_SECONDS,
                )
                return self._swap_host(url, settings.MINIO_PUBLIC_URL)
            except Exception as e:
                logger.error(f"MinIO presign failed: {e}")
                return None


minio_service = MinioService()
