#!/usr/bin/env bash
set -euo pipefail

PASS="${PASSWORD:-tigergraph}"
TG_HOME="/home/tigergraph"
GADMIN="${TG_HOME}/tigergraph/app/cmd/gadmin"
TG_CFG="${TG_HOME}/tigergraph/data/configs/tg.cfg"   # <- use data/configs

# Ensure OS password (root op; idempotent)
echo "tigergraph:${PASS}" | chpasswd || true

# Ensure tigergraph has ~/.tg.cfg pointing to the real config
ln -sf "${TG_CFG}" "${TG_HOME}/.tg.cfg"
chown -h tigergraph:tigergraph "${TG_HOME}/.tg.cfg"

# Start infra first, then all (idempotent)
su - tigergraph -c "${GADMIN} start infra" || true
su - tigergraph -c "${GADMIN} start all" || true
sleep 5

# Wait for REST++ to respond
for i in $(seq 1 120); do
  curl -fsS http://localhost:9000/echo >/dev/null 2>&1 && { echo "[tg_boot] RESTPP is up."; break; }
  sleep 2
done

# Show status for visibility
su - tigergraph -c "${GADMIN} status" || true

# Keep container alive
tail -f /home/tigergraph/tigergraph*/log/* 2>/dev/null || exec sleep infinity
