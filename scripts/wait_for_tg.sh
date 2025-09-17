#!/usr/bin/env bash
set -euo pipefail
if curl -fsS http://localhost:9000/echo >/dev/null 2>&1; then
  exit 0
else
  exit 1
fi
