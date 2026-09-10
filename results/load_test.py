#!/usr/bin/env python3
"""Load harness: fires concurrent requests at a running /triage service and reports p95
latency and cost per message. Also writes results/predictions.jsonl by replaying every
published row (dev + dev_noisy + hostile) through the ACTUAL running service -- not the
offline batch pipeline -- so the scored predictions file is exactly what the service produces.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

LLM_COST_PER_CALL_USD = 0.0015  # placeholder unit cost for when the LLM fallback is wired in;
                                 # 0 calls are made in the current cheap-path-only system.


def load_all_rows() -> list[dict]:
    rows = []
    for name in ("dev.jsonl", "dev_noisy.jsonl", "hostile.jsonl"):
        with open(f"../data/{name}", encoding="utf-8") as fh:
            rows.extend(json.loads(l) for l in fh if l.strip())
    return rows


def call(client: httpx.Client, base_url: str, row: dict) -> dict:
    payload = {"id": row["id"], "context": row["context"]}
    if row.get("turn_index") is not None:
        payload["turn_index"] = row["turn_index"]
    started = time.perf_counter()
    resp = client.post(f"{base_url}/triage", json=payload, timeout=10.0)
    elapsed_ms = (time.perf_counter() - started) * 1000
    resp.raise_for_status()
    return {"id": row["id"], "elapsed_ms": elapsed_ms, "response": resp.json()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out-predictions", default="predictions_service.jsonl")
    ap.add_argument("--out-report", default="load_test_report.json")
    args = ap.parse_args()

    rows = load_all_rows()
    print(f"replaying {len(rows)} rows at concurrency={args.concurrency}")

    latencies = []
    predictions = []
    per_request = []
    errors = 0
    t0 = time.perf_counter()
    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(call, client, args.base_url, row): row for row in rows}
            for fut in as_completed(futures):
                row = futures[fut]
                try:
                    result = fut.result()
                    latencies.append(result["elapsed_ms"])
                    predictions.append(result["response"])
                    per_request.append({"id": result["id"], "elapsed_ms": round(result["elapsed_ms"], 2)})
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    print(f"  ERROR on {row['id']}: {exc}")
    total_s = time.perf_counter() - t0

    latencies.sort()
    n = len(latencies)
    p50 = latencies[int(0.50 * (n - 1))]
    p95 = latencies[int(0.95 * (n - 1))]
    p99 = latencies[int(0.99 * (n - 1))]
    throughput = n / total_s if total_s > 0 else 0.0

    llm_calls = 0  # cheap-path-only system: no LLM fallback wired in yet
    cost_per_msg = llm_calls * LLM_COST_PER_CALL_USD / max(1, n)

    report = {
        "sample_count": n, "concurrency": args.concurrency, "errors": errors,
        "total_wall_time_s": round(total_s, 3), "throughput_rps": round(throughput, 2),
        "latency_ms": {"p50": round(p50, 2), "p95": round(p95, 2), "p99": round(p99, 2),
                        "max": round(max(latencies), 2), "min": round(min(latencies), 2)},
        "cost": {"llm_calls": llm_calls, "cost_per_message_usd": cost_per_msg,
                  "note": "Cheap-path-only system (no LLM fallback wired in). Marginal cost "
                          "is CPU time only; $0 in API spend."},
        "measured_by": "externally, via HTTP client in results/load_test.py (wall-clock per request)",
        "per_request": per_request,
    }

    Path(args.out_report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(args.out_predictions, "w", encoding="utf-8") as fh:
        for p in predictions:
            fh.write(json.dumps(p) + "\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
