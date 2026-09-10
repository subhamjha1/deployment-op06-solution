# LANDSCAPE — what already exists for this problem

One page on prior art we actually looked at before building anything, what we adopted, what we
rejected, and why. This is the "did you know what already exists" evidence for the Starter-tier
rubric.

## Hosted intent-classification APIs (considered, rejected)

Rasa NLU, Dialogflow CX, and AWS Lex all offer turnkey intent classifiers with a similar
train-on-examples workflow. We rejected all three for this task specifically, not in general:

- None of them natively encode a **conversation-stage-dependent, policy-constrained action**
  the way this task needs (intent alone predicts the majority action only 27% of the time per
  the brief — that's a second, dependent classification problem these tools aren't built for).
  We would have had to bolt a custom policy layer on top regardless, which erodes most of the
  "hosted" convenience.
- They add a network dependency and per-call pricing for a task explicitly scoped as
  "budget-class models only." A CPU-local classifier has effectively zero marginal cost.
- They're a reasonable choice if the deliverable were *just* intent classification. It isn't.

## Embedding + nearest-neighbor / sentence-transformer classifiers (considered, partially adopted in spirit)

Using `sentence-transformers` (e.g. all-MiniLM-L6-v2) embeddings + a lightweight classifier head
is a common, well-documented pattern for intent classification and would likely outperform TF-IDF
on paraphrase-heavy inputs (exactly the "screenshot description" noise type we struggle with).
We did not adopt it for this submission because:

- It requires downloading and running a ~80MB transformer model — still CPU-only and
  "budget-class," but a meaningfully heavier dependency and slower per-request than TF-IDF
  (single-digit ms vs. tens of ms), and this environment's single-core constraint makes that
  gap matter more, not less.
- TF-IDF + calibrated logistic regression already cleared all 6 bars once combined with
  policy-constrained decoding and targeted data augmentation, so the added complexity wasn't
  earning its keep for *this* submission.
- **Flagged as the first thing to try** if the screenshot/two-intents categories need further
  improvement — see ANALYSIS.md "Rejected / deferred options."

## Calibration methods: literature and what we used

The confidence-calibration bar (top-70% error ratio) is a direct pointer at model calibration
research (Platt scaling / Guo et al. 2017, "On Calibration of Modern Neural Networks", and the
older isotonic-regression calibration literature it builds on). We used scikit-learn's
`CalibratedClassifierCV(method="isotonic")`, which is the standard, well-tested implementation
of exactly this idea, rather than hand-rolling a calibration curve. Isotonic over Platt/sigmoid
because isotonic makes no parametric assumption about the shape of the miscalibration, which
matters here since we have no prior reason to expect a sigmoid-shaped bias in a linear model's
scores on this data.

## Rules engines / expert systems for the policy layer (considered, rejected as primary; used the underlying idea)

Classic expert-system-style rule engines (e.g. encoding guidelines.json as a Drools-style
decision table) were considered for the policy layer. We rejected a full rules-engine dependency
as overkill — the actual constraint (`intent -> permitted_actions`) turned out to be a flat,
verified-deterministic lookup table (we empirically confirmed permitted_actions is a pure
function of intent across all 20,000 training rows), so a plain JSON dictionary loaded at
startup captures 100% of what a rules engine would provide here, with none of the added
dependency weight. The general *idea* from that literature — keep policy as a separate,
inspectable data artifact from the model, not compiled into it — is exactly what we did with
`models/policy_table.json` and `scripts/build_policy_table.py`.

## Synthetic data augmentation for robustness

Character-level typo injection and back-translation-style paraphrase augmentation are both
well-established techniques for noise robustness (e.g. NLPAug and similar libraries implement
typo/keyboard-noise augmenters). We hand-rolled our six augmenters rather than pulling in
a library because each one targets an *exact, published* noise type from `dev_noisy.jsonl`
(typo, code_mixed, two_intents, empty, very_long, screenshot) rather than a generic noise
distribution — a general-purpose library would have covered typos well but had nothing built
for the task-specific ones (screenshot-style indirection, very-long-with-buried-signal), which
were the categories that actually mattered here.
