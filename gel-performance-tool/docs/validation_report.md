# Validation report (synthetic data, config_version 1.0.0)

Truth: 'replace' = true replacement point <= 25 cycles away or passed.

## type_A
### Held-out batches (never seen by the tree) - the honest result
- cycles compared: 1153 (no score: 0)
- cycle-level agreement: 95.1%
- balanced accuracy: 91.2%
- true 'replace' cycles caught by NO-GO: 85.3%
- NO-GO decisions that were right: 84.9%

| truth | predicted GO | predicted NO-GO |
|---|---|---|
| true: keep | 933 | 29 |
| true: replace | 28 | 163 |

| gel_batch | true_replacement_age | first_no_go_age | warning_cycles_before_true_point | outcome |
|---|---|---|---|---|
| GB-A02 | 348 | 283 | 65 | in time |
| GB-A04 | 340 | 310 | 30 | in time |
| GB-A06 | 352 | 355 | -3 | late |

### Calibration batches (the tree was trained on these - optimistic)
- cycles compared: 1022 (no score: 0)
- cycle-level agreement: 99.4%
- balanced accuracy: 99.5%
- true 'replace' cycles caught by NO-GO: 99.6%
- NO-GO decisions that were right: 98.1%

| truth | predicted GO | predicted NO-GO |
|---|---|---|
| true: keep | 761 | 5 |
| true: replace | 1 | 255 |

| gel_batch | true_replacement_age | first_no_go_age | warning_cycles_before_true_point | outcome |
|---|---|---|---|---|
| GB-A01 | 325 | 300 | 25 | in time |
| GB-A03 | 264 | 240 | 24 | in time |
| GB-A05 | 255 | 225 | 30 | in time |

## type_B
### Held-out batches (never seen by the tree) - the honest result
- cycles compared: 1116 (no score: 0)
- cycle-level agreement: 92.5%
- balanced accuracy: 91.8%
- true 'replace' cycles caught by NO-GO: 90.7%
- NO-GO decisions that were right: 75.3%

| truth | predicted GO | predicted NO-GO |
|---|---|---|
| true: keep | 837 | 64 |
| true: replace | 20 | 195 |

| gel_batch | true_replacement_age | first_no_go_age | warning_cycles_before_true_point | outcome |
|---|---|---|---|---|
| GB-B02 | 363 | 233 | 130 | too early |
| GB-B04 | 338 | 333 | 5 | in time |
| GB-B06 | 278 | 237 | 41 | in time |

### Calibration batches (the tree was trained on these - optimistic)
- cycles compared: 1105 (no score: 0)
- cycle-level agreement: 97.3%
- balanced accuracy: 96.8%
- true 'replace' cycles caught by NO-GO: 96.0%
- NO-GO decisions that were right: 92.2%

| truth | predicted GO | predicted NO-GO |
|---|---|---|
| true: keep | 837 | 20 |
| true: replace | 10 | 238 |

| gel_batch | true_replacement_age | first_no_go_age | warning_cycles_before_true_point | outcome |
|---|---|---|---|---|
| GB-B01 | 340 | 300 | 40 | in time |
| GB-B03 | 265 | 250 | 15 | in time |
| GB-B05 | 330 | 300 | 30 | in time |

