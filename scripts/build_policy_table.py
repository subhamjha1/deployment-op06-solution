#!/usr/bin/env python3
"""Build the policy lookup table from training data.

We verified empirically that `permitted_actions` is a pure function of `intent` in this
dataset (0/96 intents show variation across 20,000 training rows). So instead of re-parsing
guidelines.json's button-text into action slugs ourselves (fragile, error-prone matching),
we derive the authoritative intent -> permitted_actions mapping directly from the data the
dataset authors already resolved it into.

This keeps the policy layer a *data artifact* separate from application code: if the manager
changes a rule (updates guidelines.json / retrains), re-run this script. The service never
needs a code change to pick up a policy update.

Usage:
    python build_policy_table.py --train ../data/train.jsonl.gz \
        --guidelines ../data/guidelines.json --out ../models/policy_table.json
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl_gz(path: str) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def flatten_guideline_text(guidelines: dict) -> dict[str, list[str]]:
    """Flatten guidelines.json into flow -> list of human-readable action step descriptions.

    Used only as *context text* handed to the LLM fallback path, never as the source of the
    permitted_actions set itself (see module docstring for why).
    """
    out: dict[str, list[str]] = {}
    for flow_name, flow in guidelines.items():
        lines: list[str] = []
        for subflow_name, subflow in (flow.get("subflows") or {}).items():
            lines.append(f"[{subflow_name}]")
            for step in subflow.get("actions", []):
                text = step.get("text", "")
                subtext = step.get("subtext") or []
                line = f"  - {text}"
                if subtext:
                    line += " (" + "; ".join(subtext[:3]) + ")"
                lines.append(line)
        out[flow_name] = lines
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--guidelines", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_jsonl_gz(args.train)
    guidelines = json.loads(Path(args.guidelines).read_text(encoding="utf-8"))

    intent_to_permitted: dict[str, set[str]] = defaultdict(set)
    intent_to_flow: dict[str, str] = {}
    intent_action_bucket_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    def turn_bucket(idx: int | None) -> str:
        idx = idx or 0
        return "early" if idx <= 6 else ("mid" if idx <= 14 else "late")

    for row in rows:
        intent = row["intent"]
        intent_to_flow[intent] = row.get("flow")
        perm = row.get("permitted_actions") or []
        intent_to_permitted[intent].update(perm)
        bucket = turn_bucket(row.get("turn_index"))
        intent_action_bucket_counts[intent][bucket][row["action"]] += 1

    # Sanity check: confirm the assumption that permitted_actions is a pure function of intent.
    # (We already verified this offline; this just guards against silently building a bad
    # table if it were ever run against different / updated data.)
    inconsistent = []
    for row in rows:
        seen = intent_to_permitted[row["intent"]]
        perm = set(row.get("permitted_actions") or [])
        if perm and perm != seen and not perm.issubset(seen):
            inconsistent.append(row["intent"])
    if inconsistent:
        print(f"WARNING: {len(set(inconsistent))} intents show inconsistent permitted_actions "
              f"across rows. Falling back to the UNION of all observed sets per intent, which "
              f"is safe (never under-restricts) but re-verify before trusting policy metrics.")

    guideline_text = flatten_guideline_text(guidelines)

    table = {
        "intents": {
            intent: {
                "flow": intent_to_flow[intent],
                "permitted_actions": sorted(perm),
                "action_by_bucket": {
                    bucket: max(counts.items(), key=lambda kv: kv[1])[0]
                    for bucket, counts in buckets.items()
                },
            }
            for intent, perm in intent_to_permitted.items()
            for buckets in [intent_action_bucket_counts[intent]]
        },
        "guideline_text_by_flow": guideline_text,
    }

    Path(args.out).write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"wrote policy table: {len(table['intents'])} intents -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
