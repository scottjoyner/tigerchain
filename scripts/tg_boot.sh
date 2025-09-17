#!/usr/bin/env bash
set -euo pipefail

# Start all services (idempotent)
su - tigergraph -c "gadmin start all" || true

# Simple wait: loop until REST++ responds
for i in {1..120}; do
  if curl -fsS http://localhost:9000/echo >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

# Keep the container alive
tail -f /home/tigergraph/tigergraph*/log/* 2>/dev/null || exec sleep infinity
