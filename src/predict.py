"""End-to-end prediction pipeline (Layers 0-2, minus a live LLM call by default).

Layer 0: sanitize.py gate -> immediate abstention for hostile/empty/malformed input.
Layer 1: cheap TF-IDF + calibrated-LogisticRegression classifier (models/cheap_model.pkl),
         action prediction constrained to the intent's permitted_actions (models/policy_table.json).
Layer 2: LLM fallback for low-confidence cases. Only fires if ANTHROPIC_API_KEY is set in the
         environment; otherwise the pipeline degrades gracefully to "cheap prediction, flagged
         needs_human=True" so it is always safe to run without network/API access.

This lets us report **honest current numbers for the cheap path alone** (Layer 0+1) first, and
then separately measure how much the LLM fallback recovers once wired to a real key.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import hstack

sys.path.insert(0, str(Path(__file__).parent))
from sanitize import sanitize, abstain_response  # noqa: E402

TAIL_TURNS = 6
LOW_CONF_LLM_THRESHOLD = 0.50   # below this, we WOULD call the LLM fallback if available
NEEDS_HUMAN_THRESHOLD = 0.35    # below this (after any LLM attempt), flag for a human


def turn_bucket(idx):
    idx = idx or 0
    return "early" if idx <= 6 else ("mid" if idx <= 14 else "late")


def row_text(context: list[dict]) -> str:
    turns = (context or [])[-TAIL_TURNS:]
    return " ".join(f"[{t.get('speaker','?')}] {t.get('text','')}" for t in turns)


class Pipeline:
    def __init__(self, model_path: str, policy_path: str):
        with open(model_path, "rb") as fh:
            bundle = pickle.load(fh)
        self.word_vec = bundle["word_vec"]
        self.char_vec = bundle["char_vec"]
        self.intent_clf = bundle["intent_clf"]
        self.action_clf = bundle["action_clf"]
        self.cat_enc = bundle["cat_enc"]
        self.policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))["intents"]
        self.llm_enabled = bool(os.environ.get("ANTHROPIC_API_KEY"))

    # -- cheap path -------------------------------------------------------
    def _cheap_predict(self, context: list[dict], turn_index: int | None) -> dict:
        text = row_text(context)
        Xw = self.word_vec.transform([text])
        Xc = self.char_vec.transform([text])
        X = hstack([Xw, Xc]).tocsr()

        intent_proba = self.intent_clf.predict_proba(X)[0]
        intent_classes = self.intent_clf.classes_
        top_i = int(np.argmax(intent_proba))
        intent = intent_classes[top_i]
        intent_conf = float(intent_proba[top_i])

        bucket = turn_bucket(turn_index)
        cat = self.cat_enc.transform(np.array([[intent, bucket]], dtype=object))
        X_action = hstack([X, cat]).tocsr()
        action_proba = self.action_clf.predict_proba(X_action)[0]
        action_classes = self.action_clf.classes_

        permitted = set(self.policy.get(intent, {}).get("permitted_actions", []))
        if permitted:
            mask = np.array([a in permitted for a in action_classes])
            if mask.any():
                masked = np.where(mask, action_proba, 0.0)
                masked_sum = masked.sum()
                if masked_sum > 0:
                    masked = masked / masked_sum
                top_a = int(np.argmax(masked))
                action = action_classes[top_a]
                action_conf = float(masked[top_a])
            else:
                top_a = int(np.argmax(action_proba))
                action, action_conf = action_classes[top_a], float(action_proba[top_a])
        else:
            top_a = int(np.argmax(action_proba))
            action, action_conf = action_classes[top_a], float(action_proba[top_a])

        return {
            "intent": intent, "intent_confidence": intent_conf,
            "action": action, "action_confidence": action_conf,
        }

    # -- LLM fallback (stub-safe) ------------------------------------------
    def _llm_predict(self, row_id: str, context: list[dict], cheap: dict) -> dict | None:
        """Only called when cheap-path confidence is low. Returns None (=> fall back to the
        cheap prediction, flagged for a human) unless a real API key is configured. See
        README/MEMO for the intended prompt design once wired up."""
        if not self.llm_enabled:
            return None
        # Intentionally left as an extension point: build a prompt from `context`, the
        # candidate intent's guideline text (self.policy / guideline_text_by_flow) and the
        # permitted_actions list, call the Anthropic API, and parse a structured JSON reply.
        return None

    def predict_row(self, row: dict) -> dict:
        row_id = str(row["id"])
        result = sanitize(row.get("context") or [])
        if not result.ok:
            return abstain_response(row_id, result.reason)

        cheap = self._cheap_predict(result.cleaned_context, row.get("turn_index"))
        final_action, final_conf, used_llm = cheap["action"], cheap["action_confidence"], False

        if cheap["action_confidence"] < LOW_CONF_LLM_THRESHOLD:
            llm_result = self._llm_predict(row_id, result.cleaned_context, cheap)
            if llm_result is not None:
                final_action, final_conf, used_llm = llm_result["action"], llm_result["confidence"], True

        needs_human = final_conf < NEEDS_HUMAN_THRESHOLD

        return {
            "id": row_id,
            "intent": cheap["intent"],
            "action": final_action,
            "confidence": round(final_conf, 4),
            "needs_human": bool(needs_human),
            "_debug_used_llm": used_llm,          # stripped before writing predictions.jsonl
        }


def load_jsonl(path: str) -> list[dict]:
    import gzip
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/cheap_model.pkl")
    ap.add_argument("--policy", default="../models/policy_table.json")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pipe = Pipeline(args.model, args.policy)
    rows = []
    for p in args.inputs:
        rows.extend(load_jsonl(p))

    n_llm = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            pred = pipe.predict_row(row)
            n_llm += pred.pop("_debug_used_llm", False)
            fh.write(json.dumps(pred) + "\n")

    print(f"wrote {len(rows)} predictions to {args.out} (llm_enabled={pipe.llm_enabled}, "
          f"would-have-used-llm-or-did={n_llm})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
