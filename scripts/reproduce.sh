#!/usr/bin/env bash
# Starts the OP-06 triage service on $PORT (default 8000).
# GET /healthz returns 200 once the model is loaded. POST /triage follows the service contract
# in references/OP-06/README.md.
set -euo pipefail

PORT="${PORT:-8000}"
cd "$(dirname "$0")/../src"

exec python3 -m uvicorn service:app --host 0.0.0.0 --port "$PORT" --workers 1
