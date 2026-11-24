# Reference scope comparison (synthetic data, held-out batches only)

Each scope is calibrated on the odd-numbered batches and evaluated on the even-numbered ones.

| type | scope | calibrated ok_min / watch_min | agreement | balanced accuracy | batches in time |
|---|---|---|---|---|---|
| type_A | pooled | 65 / 31 | 83.4% | 74.4% | 2 of 3 |
| type_B | pooled | 53 / 25 | 71.9% | 64.6% | 3 of 3 |
| type_A | per_batch | 71 / 38 | 84.6% | 91.3% | 3 of 3 |
| type_B | per_batch | 59 / 17 | 85.2% | 89.7% | 3 of 3 |

Warning = cycles between the first REPLACE SOON and the true replacement point (negative = late).

| gel_batch | true replacement | warning, pooled | warning, per_batch |
|---|---|---|---|
| GB-A02 | 348 | 37 (in time) | 23 (in time) |
| GB-A04 | 340 | 9 (in time) | 34 (in time) |
| GB-A06 | 352 | -2 (late) | 25 (in time) |
| GB-B02 | 363 | 58 (in time) | 36 (in time) |
| GB-B04 | 338 | 5 (in time) | 28 (in time) |
| GB-B06 | 278 | 66 (in time) | 19 (in time) |

