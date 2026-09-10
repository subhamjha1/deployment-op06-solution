#!/usr/bin/env python3
"""Verify confidence calibration in detail: a reliability table (predicted confidence bucket
vs actual accuracy in that bucket) plus the exact top-70% error ratio the grader computes.
A well-calibrated, USEFUL confidence should show monotonically increasing accuracy across
buckets -- that's the property a manager could actually route on.
"""
import json
from collections import defaultdict

GOLD_CLEAN = "../data/dev.jsonl"
GOLD_NOISY = "../data/dev_noisy.jsonl"
PREDS = "predictions_aug.jsonl"


def load(path):
    return {json.loads(l)["id"]: json.loads(l) for l in open(path, encoding="utf-8")}


def main():
    gold = {**load(GOLD_CLEAN), **load(GOLD_NOISY)}
    preds = load(PREDS)

    rows = []
    for rid, g in gold.items():
        p = preds.get(rid)
        if p is None:
            continue
        rows.append((p["confidence"], p["action"] == g["action"]))

    rows.sort(key=lambda x: -x[0])
    n = len(rows)

    # Reliability buckets (deciles by confidence rank, matching how a manager would slice
    # "top X% most confident" in practice).
    buckets = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
               (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    print(f"{'rank decile (most->least confident)':<38}{'n':<6}{'accuracy':<10}{'conf range'}")
    reliability = []
    for lo, hi in buckets:
        i0, i1 = int(lo * n), int(hi * n)
        chunk = rows[i0:i1]
        if not chunk:
            continue
        acc = sum(ok for _, ok in chunk) / len(chunk)
        conf_range = f"{chunk[-1][0]:.3f}-{chunk[0][0]:.3f}"
        reliability.append({"decile": f"{int(lo*100)}-{int(hi*100)}%", "n": len(chunk),
                             "accuracy": round(acc, 4), "confidence_range": conf_range})
        print(f"{f'{int(lo*100)}-{int(hi*100)}% most confident':<38}{len(chunk):<6}{acc:<10.4f}{conf_range}")

    cut = max(1, int(round(0.70 * n)))
    top70 = rows[:cut]
    overall_acc = sum(ok for _, ok in rows) / n
    top70_acc = sum(ok for _, ok in top70) / len(top70)
    ratio = (1 - top70_acc) / (1 - overall_acc) if overall_acc < 1 else None

    monotonic = all(reliability[i]["accuracy"] <= reliability[i + 1]["accuracy"] + 0.05
                     for i in range(len(reliability) - 1))

    summary = {
        "n": n, "overall_accuracy": round(overall_acc, 4),
        "top70_accuracy": round(top70_acc, 4),
        "top70_error_ratio": round(ratio, 4),
        "bar": 0.85, "pass": ratio <= 0.85,
        "reliability_deciles": reliability,
        "roughly_monotonic": monotonic,
    }
    print(f"\noverall accuracy: {overall_acc:.4f}   top-70% accuracy: {top70_acc:.4f}   "
          f"error ratio: {ratio:.4f}  (bar <= 0.85 -> {'PASS' if ratio <= 0.85 else 'FAIL'})")

    with open("calibration_report.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
