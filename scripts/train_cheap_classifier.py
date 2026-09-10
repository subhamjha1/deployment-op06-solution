#!/usr/bin/env python3
"""Train the Layer-1 'cheap path' classifier: TF-IDF (word + char n-grams) -> calibrated
Logistic Regression, for both intent and (policy-constrained) action.

Char n-grams are included specifically because the noisy slice includes typos and code-mixed
text; word-level bag-of-words (what baseline.py uses) degrades badly on those, character
n-grams degrade much more gracefully.

Both classifiers are wrapped in CalibratedClassifierCV(method="isotonic") so the probability
we report is an actual calibrated P(correct), not a raw (usually overconfident) softmax value.
This directly targets the baseline's worst score: a confidence-error-ratio of 1.0 (confidence
that orders nothing).

Usage:
    python train_cheap_classifier.py --train ../data/train.jsonl.gz \
        --policy-table ../models/policy_table.json --out-dir ../models
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import re
from pathlib import Path

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

TAIL_TURNS = 6  # slightly more context than the baseline's 4, cheap to afford with TF-IDF


def load_jsonl(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def turn_bucket(idx: int | None) -> str:
    idx = idx or 0
    return "early" if idx <= 6 else ("mid" if idx <= 14 else "late")


def row_text(row: dict) -> str:
    """Text features: last TAIL_TURNS turns, speaker-tagged so the model can tell customer
    text apart from agent/action text (they carry different signal)."""
    turns = (row.get("context") or [])[-TAIL_TURNS:]
    return " ".join(f"[{t.get('speaker','?')}] {t.get('text','')}" for t in turns)


def build_features(rows: list[dict], word_vec: TfidfVectorizer, char_vec: TfidfVectorizer):
    texts = [row_text(r) for r in rows]
    Xw = word_vec.transform(texts)
    Xc = char_vec.transform(texts)
    return hstack([Xw, Xc]).tocsr()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--policy-table", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-train-rows", type=int, default=None,
                     help="subsample for faster iteration; omit to use all rows")
    ap.add_argument("--max-features", type=int, default=12000)
    ap.add_argument("--cv", type=int, default=3)
    args = ap.parse_args()

    import time
    t0 = time.time()

    rows = load_jsonl(args.train)
    if args.max_train_rows and len(rows) > args.max_train_rows:
        import random
        random.Random(0).shuffle(rows)
        rows = rows[: args.max_train_rows]
    policy = json.loads(Path(args.policy_table).read_text(encoding="utf-8"))
    texts = [row_text(r) for r in rows]

    print(f"training rows: {len(rows)}", flush=True)

    # --- Vectorizers -------------------------------------------------------
    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                token_pattern=r"(?u)\b\w+\b", max_features=args.max_features)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                sublinear_tf=True, max_features=args.max_features)
    word_vec.fit(texts)
    char_vec.fit(texts)
    X = build_features(rows, word_vec, char_vec)
    print(f"[{time.time()-t0:.1f}s] feature matrix: {X.shape}", flush=True)

    # --- Intent classifier ---------------------------------------------------
    y_intent = [r["intent"] for r in rows]
    print("training intent classifier...", flush=True)
    base_intent = LogisticRegression(max_iter=150, C=5.0, class_weight="balanced",
                                      solver="lbfgs")
    intent_clf = CalibratedClassifierCV(base_intent, method="isotonic", cv=args.cv, n_jobs=1)
    intent_clf.fit(X, y_intent)
    print(f"[{time.time()-t0:.1f}s] intent classifier fit done", flush=True)
    intent_train_acc = intent_clf.score(X, y_intent)
    print(f"  intent train accuracy (in-sample, sanity check only): {intent_train_acc:.4f}", flush=True)

    # --- Action classifier ---------------------------------------------------
    # Features: TF-IDF text + true intent one-hot (teacher-forced at train time; at inference
    # we plug in the *predicted* intent, which is standard pipeline practice) + turn-bucket
    # one-hot. Text is included because the action often depends on what the agent just asked,
    # not only on the coarse intent.
    print("training action classifier...")
    buckets = [turn_bucket(r.get("turn_index")) for r in rows]
    cat_enc = OneHotEncoder(handle_unknown="ignore")
    cat_features = cat_enc.fit_transform(
        np.array(list(zip(y_intent, buckets)), dtype=object)
    )
    X_action = hstack([X, cat_features]).tocsr()
    y_action = [r["action"] for r in rows]

    base_action = LogisticRegression(max_iter=150, C=5.0, class_weight="balanced",
                                      solver="lbfgs")
    action_clf = CalibratedClassifierCV(base_action, method="isotonic", cv=args.cv, n_jobs=1)
    action_clf.fit(X_action, y_action)
    print(f"[{time.time()-t0:.1f}s] action classifier fit done", flush=True)
    action_train_acc = action_clf.score(X_action, y_action)
    print(f"  action train accuracy (in-sample, sanity check only): {action_train_acc:.4f}", flush=True)

    # --- Persist ---------------------------------------------------------
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cheap_model.pkl", "wb") as fh:
        pickle.dump({
            "word_vec": word_vec,
            "char_vec": char_vec,
            "intent_clf": intent_clf,
            "action_clf": action_clf,
            "cat_enc": cat_enc,
            "tail_turns": TAIL_TURNS,
        }, fh)
    print(f"saved model bundle to {out / 'cheap_model.pkl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
