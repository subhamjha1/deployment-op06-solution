"""OP-06 triage service.

Implements the service contract from references/OP-06/README.md:
  - GET /healthz -> 200 once the model is loaded.
  - POST /triage -> accepts input.schema.json, returns schema.json, echoing the input id.
  - Malformed HTTP/JSON envelopes -> HTTP 400 with an error object (these are separate from
    the published valid-JSON hostile rows, which get a 200 + abstention envelope instead).
  - Bounded request timeout / body size limit, documented below, sized to comfortably fit the
    published hostile pack (longest row is a few hundred KB at most).

Run directly with:  uvicorn service:app --host 0.0.0.0 --port $PORT
(scripts/reproduce.sh wraps exactly this.)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

import sys
sys.path.insert(0, str(Path(__file__).parent))
from predict import Pipeline  # noqa: E402

MODEL_PATH = os.environ.get("MODEL_PATH", str(Path(__file__).parent.parent / "models" / "cheap_model.pkl"))
POLICY_PATH = os.environ.get("POLICY_PATH", str(Path(__file__).parent.parent / "models" / "policy_table.json"))
MAX_BODY_BYTES = 256 * 1024   # generous multiple of anything in the published hostile pack
REQUEST_TIMEOUT_S = 5.0       # bounded per-request budget; the cheap path finishes in ms

app = FastAPI(title="OP-06 Triage Service")
_pipeline: Pipeline | None = None
_loaded_at: float | None = None


class ContextTurn(BaseModel):
    speaker: str = Field(pattern="^(agent|customer|action)$")
    text: str


class TriageRequest(BaseModel):
    id: str = Field(min_length=1)
    context: list[ContextTurn] = Field(max_length=8)
    turn_index: int | None = Field(default=None, ge=0)

    class Config:
        extra = "forbid"


@app.on_event("startup")
def _load_model() -> None:
    global _pipeline, _loaded_at
    _pipeline = Pipeline(MODEL_PATH, POLICY_PATH)
    _loaded_at = time.time()


@app.get("/healthz")
def healthz():
    if _pipeline is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "loaded_at": _loaded_at}


@app.middleware("http")
async def _body_size_and_timeout_guard(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_BYTES:
        return JSONResponse(status_code=400,
                             content={"error": {"code": "payload_too_large",
                                                 "message": f"body exceeds {MAX_BODY_BYTES} bytes"}})
    import asyncio
    try:
        return await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT_S)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=400,
                             content={"error": {"code": "timeout",
                                                 "message": "request exceeded time budget"}})


@app.post("/triage")
async def triage(request: Request):
    # Parse raw JSON ourselves first: malformed transport (not valid JSON at all) is a 400,
    # distinct from a *valid* JSON envelope with hostile/garbage field values, which is a
    # normal 200 + abstention response per the contract.
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400,
                             content={"error": {"code": "invalid_json", "message": "body is not valid JSON"}})

    try:
        req = TriageRequest.model_validate(payload)
    except ValidationError as exc:
        return JSONResponse(status_code=400,
                             content={"error": {"code": "invalid_envelope", "message": str(exc)}})

    row = {"id": req.id, "context": [t.model_dump() for t in req.context],
           "turn_index": req.turn_index}
    try:
        pred = _pipeline.predict_row(row)
    except Exception as exc:  # noqa: BLE001
        # Should be unreachable given the sanitize gate, but a defensive abstention beats a 500
        # for anything the classifier layer itself throws on.
        return {"id": req.id, "intent": "unknown", "action": "none", "confidence": 0.0,
                "needs_human": True, "error": {"code": "internal_error", "message": str(exc)}}

    pred.pop("_debug_used_llm", None)
    return pred
