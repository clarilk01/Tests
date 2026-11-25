# Gel Performance Tool – 5-minute walkthrough

A script for explaining the project out loud, with the design choice behind each step and the questions
it usually raises. All data is synthetic.

---

## 0:00 – The problem (30 s)

Chromatography resin ("gel") is reused for hundreds of cycles and slowly wears out: elution pressure rises and
elution flow drops. Replacing it too early wastes expensive resin; too late risks a failed run. **This tool gives
every cycle a replacement score from 0 to 1 and a GO / NO-GO light: green = keep using, red = replace.**

## 0:30 – The data (30 s)

One row per completed cycle: gel batch, chromatography type, gel age, elution flow and elution pressure
(PT 80th percentile). I wrote a simulator with realistic problems on purpose: batch-to-batch differences, noise,
interrupted runs with missing values, sensor spikes, and two column types with very different baselines. It also
stores a hidden "true" replacement point, which I use only for training labels and validation.

> *Why synthetic?* I can show the whole method without any confidential data, and I have a ground truth to test
> against, which real data often does not.

## 1:00 – Features (1 min)

- **Rolling mean** of each KPI over the last 10 cycles of the same gel batch, so one noisy run does not flip the decision.
- **z-scores against early-life resin** (first 30 cycles of each batch), **per chromatography type**: type_B runs at
  about twice the pressure of type_A, so a shared scale would make every type_B cycle look worn.
- **Oriented so higher = worse:** pressure as it is, flow flipped.
- **Gel age** scaled as "every 100 cycles = +1". **Gel batch** groups the rolling means but is *not* a model input,
  because a tree would memorise batch names.
- Missing values are never filled in; an affected cycle is flagged.

## 2:00 – Score and decision (1 min)

- A **decision tree** of depth 3 turns the three z-scores into a score from 0 to 1: the share of similar training
  cycles that were "replace" cases. I can print the whole tree as a few if/else lines, so every score is explainable,
  and the output names the exact rule each cycle followed.
- **NO-GO only if both are true:** the rolling elution pressure is at or above the type's pressure limit, **and** the
  score is 0.5 or more. Two independent signals have to agree before anyone replaces a column.
- Trained once with scikit-learn, saved as **readable JSON**; daily scoring needs no ML library.
- Everything tunable is in `config.yaml`, and the config and model versions are written into every output row.

## 3:00 – Built like a production job (45 s)

- One command scores a whole input file and logs every run (what was read, incomplete cycles, decisions, errors).
- **Idempotent** (same input → byte-identical output; tested), atomic writes, clear exit codes.
- 46 pytest tests, mostly tiny hand-checkable examples, plus end-to-end tests.
- The Streamlit dashboard **only reads** the job's output: score and GO / NO-GO on the side, flow, pressure and score
  against time, and hovering shows the score at any moment.

## 3:45 – Does it work? (1 min)

- **No peeking:** the tree was trained on the odd-numbered batches; results come from the even-numbered ones it never saw.
- **Held-out:** 92–95% of cycles agree with the truth; balanced accuracy about 91%.
- **What I found and kept visible:**
  - **One sensor spike can cause a false NO-GO:** it lifts the 10-cycle rolling mean over the limit for exactly 10
    cycles. That happened in 2 of the 6 held-out batches. A rolling median would fix it, and that is my next change.
  - A lot that **starts** at low pressure reaches a fixed limit late (one batch 3 cycles late). Fixed limits treat
    every lot the same, which is a process decision, not a data decision.
  - On the training batches it scores 97–99%; only the held-out numbers count.

## 4:45 – Close (15 s)

"With real data I would use the real replacement history, with the reason for each replacement, as the truth;
agree the KPI definitions and pressure limits per column type with process experts; and keep the config, model and
baseline under version control."

---

## Questions you may be asked

**Why a decision tree and not a bigger model?**
There are few replacement events (one per batch), so a complex model would overfit. Operators also need to see
*why*. A depth-3 tree is a handful of readable rules, and each output row shows the rule that fired.

**Why both a pressure limit and a score?**
The limit is a simple, physical rule operators already trust. The score adds context (flow, age). Requiring both
reduces false alarms; the reason text also flags the in-between cases ("score high but pressure below limit –
keep watching").

**Why is the score in steps rather than smooth?**
A tree has a few final rules (leaves) and each has one score. A smooth score would need a different model (e.g.
logistic regression) and would be harder to explain.

**How do you stop the tree from cheating?**
It is trained only on calibration batches, the label comes from the simulator's truth, and results are reported on
held-out batches. I also show the training-batch results, to make the gap visible.

**What happens when a run is interrupted?**
The missing value stays missing. The rolling mean averages the other cycles in the window and the cycle is flagged
*incomplete*. If the whole window is missing, there is no score.

**How do you know the code is right?**
Known-answer tests with hand-worked examples (e.g. pressures 1, 2, 3 → mean 2, std 1, so 5 → z = +3), a hand-made tree
to test the predictor and every GO / NO-GO case, and end-to-end tests that prove the job is idempotent.

**What is the biggest weakness?**
The ground truth. Here I invented it; in reality it is "when did we replace the resin, and why". And the hydraulic
KPIs cannot see quality-driven failures such as yield or purity loss.
