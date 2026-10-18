#!/usr/bin/env bash
# TigerGraph Community Edition container boot. No implicit schema or license changes.
set -euo pipefail

TG_HOME="/home/tigergraph"
GADMIN="${TG_HOME}/tigergraph/app/cmd/gadmin"
TG_CFG="${TG_HOME}/tigergraph/data/configs/tg.cfg"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "[tigerchain] expected root bootstrap to switch to tigergraph service user" >&2
  exit 1
fi
if [[ ! -e "${TG_CFG}" ]]; then
  echo "[tigerchain] missing TigerGraph config: persistent data volume not initialized" >&2
  exit 1
fi

# Startup in an isolated instance only. Startup is intentionally separate
# from GSQL schema/install; never change a production graph as a side effect.
su - tigergraph -c "${GADMIN} start infra"
su - tigergraph -c "${GADMIN} start all"

ready=0
for _ in $(seq 1 120); do
  if curl -fsS --max-time 4 http://127.0.0.1:9000/echo >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "${ready}" != "1" ]]; then
  echo "[tigerchain] REST++ never reached liveness within bounded startup window" >&2
  exit 1
fi

echo "[tigerchain] REST++ alive; query/schema readiness must be tested separately"
exec sleep infinity
