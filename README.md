# OP-06 — Triage the Inbox

A `POST /triage` service that classifies a customer-support conversation prefix into
`{intent, action, confidence, needs_human}`, where `action` is constrained to what the
written procedure (`guidelines.json`) actually permits for that intent.

**Status: frozen for submission.** No LLM fallback is wired in — the cheap path (TF-IDF +
calibrated logistic regression, trained on augmented data) clears all 6 published qualification
bars on its own. See [MEMO.md](MEMO.md) for the manager-facing summary and
[ANALYSIS.md](ANALYSIS.md) for the full technical writeup.

## Repository layout

```
data/           published inputs (train/dev/dev_noisy/hostile, schemas, guidelines.json)
models/         frozen model artifact (cheap_model.pkl) + policy_table.json
src/            service.py (FastAPI app), predict.py (pipeline), sanitize.py (Layer 0 gate)
scripts/        build_policy_table.py, train_cheap_classifier.py, augment_train.py, reproduce.sh
results/        graded predictions, manifest.json, raw/ run records, analysis artifacts
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12, no GPU required. Single-core CPU is sufficient (see latency note below).

## Run the service

```bash
PORT=8000 bash scripts/reproduce.sh
```

This starts `POST /triage` and `GET /healthz` on `$PORT` (default 8000), loading the frozen
model at `models/cheap_model.pkl`. It does **not** retrain anything — the model is committed
as-is for reproducibility. See "Regenerating the model" below if you want to retrain from
scratch.

```bash
curl http://127.0.0.1:8000/healthz
curl -X POST http://127.0.0.1:8000/triage -H "Content-Type: application/json" \
  -d '{"id":"t1","context":[{"speaker":"customer","text":"i want to return an item"}]}'
```

## Evaluating

```bash
cd data
python3 grader.py --pred ../results/predictions_service.jsonl \
  --gold dev.jsonl --noisy dev_noisy.jsonl --hostile hostile.jsonl \
  --schema schema.json --report ../results/report_from_service.json
```

`results/predictions_service.jsonl` was produced by replaying every published row through the
**live running service** (`results/load_test.py`), not an offline shortcut — see
`results/manifest.json` for the run record and hashes.

## Regenerating the model (optional — not required to run the service)

```bash
cd scripts
python3 build_policy_table.py --train ../data/train.jsonl.gz \
  --guidelines ../data/guidelines.json --out ../models/policy_table.json
python3 augment_train.py --train ../data/train.jsonl.gz \
  --out ../data/train_augmented.jsonl.gz --per-type 2000 --seed 0
python3 train_cheap_classifier.py --train ../data/train_augmented.jsonl.gz \
  --policy-table ../models/policy_table.json --out-dir ../models \
  --max-features 8000 --cv 2
```

Training is CPU-only and takes ~3.5 minutes on a single core (20k→32k rows, TF-IDF word+char
n-grams, 2-fold isotonic calibration for both intent and action classifiers). Fixed seed (0)
for the augmentation step; scikit-learn's `LogisticRegression`/`CalibratedClassifierCV` are
otherwise deterministic given fixed folds.

## Load / latency testing

```bash
# with the service already running on :8000
cd results
python3 load_test.py --base-url http://127.0.0.1:8000 --concurrency 16 \
  --out-predictions predictions_service.jsonl --out-report latency_bench.json
```

**Note on the concurrency=16 numbers:** this development environment has a single CPU core.
Throughput at concurrency=16 (50.5 rps) is only modestly higher than concurrency=1 (44.1 rps) —
nowhere near the ~16x a multi-core deployment would show — which indicates the elevated p95
(368ms vs 26ms) is substantially single-core request queuing, not per-request model cost. See
`results/manifest.json` → `runs[0].notes` and `results/raw/dev-run-001_latency_bench_concurrency1.json`
for both runs side by side.

## Hostile-pack runtime verification

`grader.py`'s schema-validity check only proves the *response shape* is valid; it cannot prove
absence of crashes/hangs/injection-compliance from a predictions file alone. `results/verify_hostile_runtime.py`
drives the actual pipeline object per hostile row with a wall-clock timeout and an explicit
injection-compliance check:

```bash
cd results
python3 verify_hostile_runtime.py
```

## Qualification bars (final, frozen model)

| Metric | Bar | Value | Status |
|---|---|---|---|
| Intent accuracy | ≥ 0.55 | 0.6196 | PASS |
| Action accuracy | ≥ 0.55 | 0.7496 | PASS |
| Intent macro-F1 | ≥ 0.50 | 0.6207 | PASS |
| Clean→noisy action gap | ≤ 0.10 | 0.0992 | PASS |
| Hostile pack schema-valid | 100% | 100% | PASS |
| Hostile pack runtime (crash/hang/injection) | 0 | 0/0/0 | PASS |
| Confidence error ratio (top-70%) | ≤ 0.85 | 0.6054 | PASS |

All 6 published bars pass. Full breakdown, failure taxonomy, and what was tried and rejected:
[ANALYSIS.md](ANALYSIS.md). Decision history: [DECISIONS.md](DECISIONS.md) /
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). Existing-tools review: [LANDSCAPE.md](LANDSCAPE.md).

## Known limitation / explicit non-goal for this submission

No LLM fallback is implemented. `src/predict.py`'s `Pipeline._llm_predict()` is a documented,
inert extension point — the cheap path alone clears every bar, so the added cost/complexity/
latency of an LLM call was not justified for this submission. Per-noise-type breakdown shows
`typo`/`code_mixed`/`two_intents` regressed slightly (−0.04 each) under data augmentation; an
LLM fallback targeted at exactly those categories is the natural next experiment (see
ANALYSIS.md → "Rejected / deferred options").
