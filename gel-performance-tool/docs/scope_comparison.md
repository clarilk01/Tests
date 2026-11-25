# Reference scope comparison (synthetic data, held-out batches only)

For each scope the tree is re-trained on the calibration batches and evaluated on the held-out ones.

| type | scope | agreement | balanced accuracy | batches in time |
|---|---|---|---|---|
| type_A | pooled | 95.1% | 91.2% | 2 of 3 |
| type_B | pooled | 92.5% | 91.8% | 2 of 3 |
| type_A | per_batch | 95.2% | 87.9% | 2 of 3 |
| type_B | per_batch | 95.9% | 93.9% | 2 of 3 |

Warning = cycles between the first NO-GO and the true replacement point (negative = late).

| gel_batch | true replacement | warning, pooled | warning, per_batch |
|---|---|---|---|
| GB-A02 | 348 | 65 (in time) | 65 (in time) |
| GB-A04 | 340 | 30 (in time) | 26 (in time) |
| GB-A06 | 352 | -3 (late) | -3 (late) |
| GB-B02 | 363 | 130 (too early) | 130 (too early) |
| GB-B04 | 338 | 5 (in time) | 5 (in time) |
| GB-B06 | 278 | 41 (in time) | 34 (in time) |

