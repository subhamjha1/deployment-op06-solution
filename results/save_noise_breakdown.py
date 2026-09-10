#!/usr/bin/env python3
"""Persist the per-noise-type breakdown that grader.py does NOT report on its own (it only
reports aggregate clean vs noisy accuracy). This is required evidence for the brief's
'Required Analysis' section: noisy slice broken out by noise type.
"""
import json
from collections import defaultdict
from pathlib import Path

GOLD = "../data/dev_noisy.jsonl"
PREDS_BEFORE = "predictions.jsonl"          # pre-augmentation cheap-path run
PREDS_AFTER = "predictions_aug.jsonl"       # frozen / current model
OUT = "noise_type_breakdown.json"


def load(path):
    return {json.loads(l)["id"]: json.loads(l) for l in open(path, encoding="utf-8")}


def breakdown(gold, preds):
    by_kind_action = defaultdict(lambda: [0, 0])
    by_kind_intent = defaultdict(lambda: [0, 0])
    for rid, g in gold.items():
        p = preds.get(rid, {})
        kind = g.get("noise_kind")
        by_kind_action[kind][1] += 1
        by_kind_action[kind][0] += int(p.get("action") == g.get("action"))
        by_kind_intent[kind][1] += 1
        by_kind_intent[kind][0] += int(p.get("intent") == g.get("intent"))
    return {
        kind: {
            "n": by_kind_action[kind][1],
            "intent_accuracy": round(by_kind_intent[kind][0] / by_kind_intent[kind][1], 4),
            "action_accuracy": round(by_kind_action[kind][0] / by_kind_action[kind][1], 4),
        }
        for kind in by_kind_action
    }


def main():
    gold = load(GOLD)
    before = breakdown(gold, load(PREDS_BEFORE))
    after = breakdown(gold, load(PREDS_AFTER))

    report = {"noise_types": {}}
    for kind in before:
        report["noise_types"][kind] = {
            "n": before[kind]["n"],
            "before_augmentation": {
                "intent_accuracy": before[kind]["intent_accuracy"],
                "action_accuracy": before[kind]["action_accuracy"],
            },
            "after_augmentation": {
                "intent_accuracy": after[kind]["intent_accuracy"],
                "action_accuracy": after[kind]["action_accuracy"],
            },
            "action_accuracy_delta": round(
                after[kind]["action_accuracy"] - before[kind]["action_accuracy"], 4),
        }
    Path(OUT).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
