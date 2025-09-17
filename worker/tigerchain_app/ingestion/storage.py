from __future__ import annotations

from pathlib import Path
from typing import Tuple

import boto3
from botocore.config import Config

from ..config import Settings
from ..utils.logging import get_logger
from .pipeline import ObjectStore

logger = get_logger(__name__)


class MinioObjectStore(ObjectStore):
    """S3-compatible object storage backed by MinIO."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.minio_endpoint,
            aws_access_key_id=settings.minio_access_key,
            aws_secret_access_key=settings.minio_secret_key,
            use_ssl=settings.minio_secure,
            config=Config(s3={"addressing_style": "path"}),
            region_name="us-east-1",
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.settings.minio_bucket)
        except Exception:
            logger.info("Creating MinIO bucket %s", self.settings.minio_bucket)
            self.client.create_bucket(Bucket=self.settings.minio_bucket)

    def upload(self, path: Path, key: str) -> Tuple[str, str]:
        self.client.upload_file(str(path), self.settings.minio_bucket, key)
        uri = f"s3://{self.settings.minio_bucket}/{key}"
        http_url = f"{self.settings.minio_endpoint.rstrip('/')}/{self.settings.minio_bucket}/{key}"
        return uri, http_url
