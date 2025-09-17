#!/usr/bin/env bash
set -euo pipefail

PASS="${PASSWORD:-tigergraph}"

# Ensure OS password for 'tigergraph' (idempotent)
echo "tigergraph:${PASS}" | chpasswd || true

# --- TigerGraph env bootstrap (idempotent) ---
# 1) ~/.tg.cfg symlink for tigergraph user
ln -sf /home/tigergraph/tigergraph/configs/tg.cfg /home/tigergraph/.tg.cfg
chown -h tigergraph:tigergraph /home/tigergraph/.tg.cfg

# 2) PATH for login shells
if ! grep -q "/home/tigergraph/tigergraph/app/cmd" /home/tigergraph/.bashrc 2>/dev/null; then
  echo 'export PATH=$PATH:/home/tigergraph/tigergraph/app/cmd' >> /home/tigergraph/.bashrc
  chown tigergraph:tigergraph /home/tigergraph/.bashrc
fi

# Start all services via login shell
su - tigergraph -c "gadmin start all" || true
sleep 5

# If REST++ isn't up, try a repair once
if ! curl -fsS http://localhost:9000/echo >/dev/null 2>&1; then
  echo "[tg_boot] RESTPP not ready, attempting gadmin repair..."
  su - tigergraph -c "gadmin repair all" || true
  su - tigergraph -c "gadmin start all" || true
fi

# Wait for REST++ (up to 4 minutes)
for i in $(seq 1 120); do
  if curl -fsS http://localhost:9000/echo >/dev/null 2>&1; then
    echo "[tg_boot] RESTPP is up."
    break
  fi
  sleep 2
done

# Show status (for logs/visibility)
su - tigergraph -c "gadmin status" || true

# Keep container alive
tail -f /home/tigergraph/tigergraph*/log/* 2>/dev/null || exec sleep infinity
