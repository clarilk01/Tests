# Gel Performance Tool – 5-minute walkthrough

A script for explaining the project out loud, with the design choice behind each step and the questions
it usually raises. All data is synthetic.

---

## 0:00 – The problem (30 s)

Chromatography resin ("gel") is reused for hundreds of cycles and slowly wears out: back-pressure rises, flow drops.
Replacing it too early wastes expensive resin; too late risks a failed or out-of-spec run. Operators usually judge
from several signals at once. **This tool turns those signals into one 0–100 health score per cycle, plus an
OK / WATCH / REPLACE SOON recommendation that explains itself.**

## 0:30 – The data (30 s)

One row per completed cycle: gel batch, chromatography type, resin age, two pressure features, two flow features.
I wrote a simulator with realistic problems on purpose: batch-to-batch differences, noise, 3% interrupted runs with
missing values, sensor spikes, and two column types with very different baselines. The simulator also stores a
hidden "true" replacement point that the tool never sees. I only use it for validation.

> *Why synthetic?* It lets me show the whole method end-to-end without any confidential data, and it gives me a
> ground truth to test against, which real data often does not have.

## 1:00 – Step 1: put every feature on the same scale (1 min)

- Each feature becomes a **z-score against early-life resin**: the first 30 cycles of each batch are "what fresh
  looks like".
- **Per chromatography type**, because type_B runs at about twice the pressure of type_A. Standardising them
  together would make every type_B cycle look worn.
- **Oriented so higher = worse:** pressures as they are, flows flipped. After this, z = +2 means "two spreads worse
  than fresh" for every feature.
- **Resin age is special:** its early-life spread is tiny, so a normal z-score would explode. I use "every 100
  cycles = +1".
- **Guard rails:** z is capped at ±6 so one spike cannot dominate; missing stays missing.

## 2:00 – Step 2: one score that explains itself (1 min)

```
points lost per feature = 100 × weight × z / 4
score = 100 − total points lost   (capped to 0–100)
```

- Fresh resin = 100. Each feature takes off points, so the **contribution chart is just the points lost**. That is
  what makes the recommendation explainable: "REPLACE SOON, mainly because of displacement pressure (−19 pts)".
- **Missing feature:** its weight goes to 0, the others are scaled back up to 1, and the cycle is flagged
  *incomplete*. Too few features → no score, flagged *insufficient*. Never filled in.
- **Smoothing:** the recommendation uses the median of the last 5 scores, so one noisy cycle cannot flip it.
- **Everything tunable is in `config.yaml`**, per type, with a version number written into every output row.

## 3:00 – Step 3: built like a production job (45 s)

- One command, schedulable hourly; logs what was read, how many cycles were incomplete, and any errors.
- **Idempotent:** same input → byte-identical output (tested, even with shuffled rows). Atomic writes, so a crash
  never leaves a half-written file. Exit codes tell the scheduler what failed.
- 57 pytest tests, mostly tiny known-answer examples I can check by hand, plus an end-to-end run.
- The Streamlit dashboard **only reads** the job's output, so it can never disagree with it.

## 3:45 – Step 4: does it work? (1 min)

- **No peeking:** thresholds were chosen on the odd-numbered batches only; results are reported on the even-numbered
  batches.
- **Measured with balanced accuracy**, because most cycles are "OK" and plain accuracy would reward always saying OK.
- **Results (pooled baseline):** 72–83% cycle agreement; 5 of 6 held-out batches warned in time; it never confused
  OK with REPLACE SOON.
- **Things I found and kept visible:**
  - Calibrating on only 3 batches made type_A slightly *worse* on held-out data. That is overfitting, so I report it.
  - Two batches that *started* at high pressure were flagged early with the pooled baseline. I added a switchable
    per-batch baseline: about 90% balanced accuracy, all 6 in time. But the simulator's truth is itself defined per
    batch, so that comparison is partly built in. Which baseline is right is a process question, not a data
    science one.

## 4:45 – Close (15 s)

"With real data I would start with why data goes missing, agree the KPI definitions per column type with process
experts, use real past replacement decisions as the truth, and keep the config and the baseline under version
control."

---

## Questions you may be asked

**Why not a machine-learning model?**
There are few replacement events (one per batch), so a supervised model has little to learn from. Operators also
need to see *why*. A transparent weighted score with calibrated thresholds is easier to validate, explain and
change-control. ML would make sense once there are many labelled replacements.

**How did you choose the weights?**
By hand: pressures highest, then flows, then age. Only the thresholds are calibrated. With real data I would check
the weights with experts and test their sensitivity, e.g. how often the recommendation changes if a weight moves by
±0.05.

**What if a new chromatography type is added?**
Add a section to `config.yaml`. The loader refuses to start if a feature, weight or threshold is missing. The tool
needs at least 2 early-life cycles per feature to build a baseline, and the thresholds should be calibrated
before anyone relies on them.

**What happens when a run is interrupted?**
Missing features stay missing. The cycle is scored from what remains with re-normalised weights and flagged
*incomplete*, or not scored at all below 3 features. The log counts them every run.

**Why the median for smoothing, not the mean?**
A median ignores one extreme cycle completely; a mean would be pulled by it. The cost is a lag of 2–3 cycles.

**How do you know the code is right?**
Known-answer tests: tiny examples worked out by hand (e.g. pressures 1, 2, 3 → mean 2, std 1, so 5 → z = +3), one
test per rule (each feature's direction, weight re-normalisation, every threshold boundary), and an end-to-end test
that also proves idempotency.

**What is the biggest weakness?**
The ground truth. Here I invented it; in reality it is "when did we replace the resin, and why". Without it you can
only show the score is consistent, not that it is correct. Also, these features only see hydraulics. Resin can fail
on yield or purity first.
