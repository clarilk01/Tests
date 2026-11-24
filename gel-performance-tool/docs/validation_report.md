# Validation report (synthetic data, config_version 0.2.1)

## type_A
Calibrated on ['GB-A01', 'GB-A03', 'GB-A05']: ok_min = 65, watch_min = 31

### Held-out batches ['GB-A02', 'GB-A04', 'GB-A06'] - config thresholds (ok_min 65, watch_min 31)
- cycles compared: 1150 (no score: 3)
- cycle-level agreement: 83.4%
- balanced accuracy: 74.4%

Confusion (rows = true class, columns = predicted):

| true \ predicted | OK | WATCH | REPLACE SOON |
|---|---|---|---|
| OK | 728 | 82 | 0 |
| WATCH | 60 | 83 | 6 |
| REPLACE SOON | 0 | 43 | 148 |

| gel_batch | true_replacement_age | first_replace_soon_age | warning_cycles_before_true_point | outcome |
|---|---|---|---|---|
| GB-A02 | 348 | 311 | 37 | in time |
| GB-A04 | 340 | 331 | 9 | in time |
| GB-A06 | 352 | 354 | -2 | late |

### Held-out batches ['GB-A02', 'GB-A04', 'GB-A06'] - placeholder 70 / 40 (ok_min 70, watch_min 40)
- cycles compared: 1150 (no score: 3)
- cycle-level agreement: 81.7%
- balanced accuracy: 77.0%

Confusion (rows = true class, columns = predicted):

| true \ predicted | OK | WATCH | REPLACE SOON |
|---|---|---|---|
| OK | 685 | 125 | 0 |
| WATCH | 40 | 92 | 17 |
| REPLACE SOON | 0 | 29 | 162 |

| gel_batch | true_replacement_age | first_replace_soon_age | warning_cycles_before_true_point | outcome |
|---|---|---|---|---|
| GB-A02 | 348 | 308 | 40 | in time |
| GB-A04 | 340 | 313 | 27 | in time |
| GB-A06 | 352 | 342 | 10 | in time |

## type_B
Calibrated on ['GB-B01', 'GB-B03', 'GB-B05']: ok_min = 53, watch_min = 25

### Held-out batches ['GB-B02', 'GB-B04', 'GB-B06'] - config thresholds (ok_min 53, watch_min 25)
- cycles compared: 1109 (no score: 7)
- cycle-level agreement: 71.9%
- balanced accuracy: 64.6%

Confusion (rows = true class, columns = predicted):

| true \ predicted | OK | WATCH | REPLACE SOON |
|---|---|---|---|
| OK | 563 | 185 | 0 |
| WATCH | 44 | 41 | 64 |
| REPLACE SOON | 0 | 19 | 193 |

| gel_batch | true_replacement_age | first_replace_soon_age | warning_cycles_before_true_point | outcome |
|---|---|---|---|---|
| GB-B02 | 363 | 305 | 58 | in time |
| GB-B04 | 338 | 333 | 5 | in time |
| GB-B06 | 278 | 212 | 66 | in time |

### Held-out batches ['GB-B02', 'GB-B04', 'GB-B06'] - placeholder 70 / 40 (ok_min 70, watch_min 40)
- cycles compared: 1109 (no score: 7)
- cycle-level agreement: 56.8%
- balanced accuracy: 58.1%

Confusion (rows = true class, columns = predicted):

| true \ predicted | OK | WATCH | REPLACE SOON |
|---|---|---|---|
| OK | 388 | 283 | 77 |
| WATCH | 8 | 41 | 100 |
| REPLACE SOON | 0 | 11 | 201 |

| gel_batch | true_replacement_age | first_replace_soon_age | warning_cycles_before_true_point | outcome |
|---|---|---|---|---|
| GB-B02 | 363 | 226 | 137 | too early |
| GB-B04 | 338 | 321 | 17 | in time |
| GB-B06 | 278 | 136 | 142 | too early |

