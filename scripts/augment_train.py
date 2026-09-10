#!/usr/bin/env python3
"""Targeted synthetic-noise augmentation, one generator per noise type actually observed in
dev_noisy.jsonl (confirmed via breakdown: typo, code_mixed, two_intents, empty, very_long,
screenshot -- 50 each, 300 total).

Design constraints (as agreed):
  - Original clean rows are always preserved untouched.
  - Augmented rows are added at a controlled ratio, not blown up into a huge dataset.
  - Each augmented row keeps the SAME intent/action/permitted_actions/turn_index as its source
    row -- we are only perturbing surface form, not changing the label.

Usage:
    python augment_train.py --train ../data/train.jsonl.gz --out ../data/train_augmented.jsonl.gz \
        --per-type 2000 --seed 0
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import random
from pathlib import Path

# A small pool of Hinglish/code-mixed connector phrases, drawn from the style actually seen
# in the published dev_noisy.jsonl sample (e.g. "thoda jaldi karo please").
CODE_MIX_PHRASES = [
    "thoda jaldi karo please", "please jaldi karna", "bhai yeh dikkat hai",
    "koi baat nahi, thanks", "ek dum sahi", "mujhe abhi chahiye please",
    "yeh toh bahut frustrating hai", "please help karo jaldi",
]

# Filler/complaint sentences used to inflate a message the way the real "very_long" noise does
# (a real customer ask buried inside a long complaint), never the intent-bearing tokens.
LONG_FILLER = [
    "i have been waiting for a very long time and this is extremely frustrating",
    "this is taking forever and i really need this resolved as soon as possible",
    "i have already explained this multiple times to other agents and nothing has happened",
    "i am a long time customer and i expect much better service than this honestly",
    "this whole process has been so confusing and nobody seems to know what is going on",
    "i really hope someone can actually help me today because i am losing my patience",
]

# Screenshot-style descriptions: a visual symptom, no direct topical keyword overlap with any
# specific intent -- mirrors the real noise type, which forces reliance on surrounding context
# rather than the noisy turn itself.
SCREENSHOT_TEMPLATES = [
    "i am attaching a screenshot. it shows a red banner at the top and a spinning circle that never stops.",
    "here is a screenshot of what i am seeing, there is an error icon and nothing happens when i click it.",
    "attaching a screenshot - the page just shows a blank grey box where the button should be.",
    "sending a screenshot, it shows a small popup with an exclamation mark and no other text.",
    "screenshot attached, the screen just keeps loading and nothing else shows up.",
]

VOWELS = "aeiou"


def typo_noise(text: str, rng: random.Random) -> str:
    words = text.split(" ")
    n_hits = max(1, int(len(words) * 0.15))
    idxs = rng.sample(range(len(words)), min(n_hits, len(words)))
    for i in idxs:
        w = words[i]
        if len(w) < 3:
            continue
        op = rng.choice(["swap", "delete", "dup", "vowel"])
        pos = rng.randrange(1, len(w) - 1)
        if op == "swap":
            w = w[:pos] + w[pos + 1] + w[pos] + w[pos + 2:]
        elif op == "delete":
            w = w[:pos] + w[pos + 1:]
        elif op == "dup":
            w = w[:pos] + w[pos] + w[pos:]
        elif op == "vowel" and w[pos].lower() in VOWELS:
            w = w[:pos] + rng.choice(VOWELS) + w[pos + 1:]
        words[i] = w
    return " ".join(words)


def code_mixed_noise(text: str, rng: random.Random) -> str:
    phrase = rng.choice(CODE_MIX_PHRASES)
    return f"{text} {phrase}" if rng.random() < 0.5 else f"{phrase}, {text}"


def two_intents_noise(text: str, other_text: str, rng: random.Random) -> str:
    connector = rng.choice([" also i want to know ", " and also ", " oh and by the way "])
    return f"{text}{connector}{other_text.strip().rstrip('.').lower()}"


def empty_noise(_text: str, _rng: random.Random) -> str:
    return ""


def very_long_noise(text: str, rng: random.Random) -> str:
    n_filler = rng.randint(20, 60)  # inflates to roughly the multi-thousand-char range observed
    filler = " ".join(rng.choice(LONG_FILLER) for _ in range(n_filler))
    return f"{text}. {filler}" if rng.random() < 0.5 else f"{filler}. {text}"


def screenshot_noise(_text: str, rng: random.Random) -> str:
    # Replaces (not appends to) the turn, matching the real pattern: the visual description
    # stands in for the actual ask, and any surviving signal must come from OTHER turns in
    # the conversation -- which augmentation correctly leaves untouched.
    return rng.choice(SCREENSHOT_TEMPLATES)


def pick_customer_turn_idx(context: list[dict], rng: random.Random, prefer_last: bool = True) -> int | None:
    cust_idxs = [i for i, t in enumerate(context) if t.get("speaker") == "customer"]
    if not cust_idxs:
        return None
    return cust_idxs[-1] if prefer_last else rng.choice(cust_idxs)


def make_augmented(row: dict, kind: str, rng: random.Random, other_row: dict | None = None) -> dict | None:
    context = copy.deepcopy(row.get("context") or [])
    idx = pick_customer_turn_idx(context, rng, prefer_last=(kind != "screenshot"))
    if idx is None:
        return None
    text = context[idx]["text"]

    if kind == "typo":
        context[idx]["text"] = typo_noise(text, rng)
    elif kind == "code_mixed":
        context[idx]["text"] = code_mixed_noise(text, rng)
    elif kind == "two_intents":
        if other_row is None:
            return None
        other_idx = pick_customer_turn_idx(other_row.get("context") or [], rng)
        if other_idx is None:
            return None
        other_text = other_row["context"][other_idx]["text"]
        context[idx]["text"] = two_intents_noise(text, other_text, rng)
    elif kind == "empty":
        context[idx]["text"] = empty_noise(text, rng)
    elif kind == "very_long":
        context[idx]["text"] = very_long_noise(text, rng)
    elif kind == "screenshot":
        # Pick a *non-final* customer turn when possible so the informative turns survive,
        # matching the real examples (the ask is often stated earlier, the screenshot line
        # is a later, less informative turn).
        context[idx]["text"] = screenshot_noise(text, rng)
    else:
        raise ValueError(kind)

    new_row = copy.deepcopy(row)
    new_row["context"] = context
    new_row["id"] = f"{row['id']}-aug-{kind}"
    return new_row


def load_jsonl_gz(path: str) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-type", type=int, default=2000,
                     help="augmented rows to generate per noise type")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = load_jsonl_gz(args.train)
    print(f"source rows: {len(rows)}")

    kinds = ["typo", "code_mixed", "two_intents", "empty", "very_long", "screenshot"]
    augmented: list[dict] = []
    for kind in kinds:
        sample = rng.sample(rows, min(args.per_type, len(rows)))
        made = 0
        for row in sample:
            other = rng.choice(rows) if kind == "two_intents" else None
            new_row = make_augmented(row, kind, rng, other_row=other)
            if new_row is not None:
                augmented.append(new_row)
                made += 1
        print(f"  {kind:<14} +{made}")

    all_rows = rows + augmented
    rng.shuffle(all_rows)
    print(f"total rows after augmentation: {len(all_rows)} "
          f"({len(augmented)} added, {len(augmented)/len(rows)*100:.1f}% growth)")

    with gzip.open(args.out, "wt", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
