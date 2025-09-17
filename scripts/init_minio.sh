#!/usr/bin/env sh
set -eu
: "${MINIO_ENDPOINT:=http://minio-tg:9000}"
MC_ALIAS="local"
mc alias set "$MC_ALIAS" "$MINIO_ENDPOINT" "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"
mc mb --ignore-existing "$MC_ALIAS/$MINIO_BUCKET"
mc anonymous set download "$MC_ALIAS/$MINIO_BUCKET" || true
echo "[init_minio] Bucket ensured: $MINIO_BUCKET at $MINIO_ENDPOINT"
