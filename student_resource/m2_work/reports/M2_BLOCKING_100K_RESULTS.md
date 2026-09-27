# M2 — Blocking & Candidate Retrieval


## Stage 1 — Controlled 100K Experiment


### Configuration

- Random seed: `42`

- Source 1 sample: **100,000**

- Batch size: **5,000**


### Blocking Strategies

- `name_exact`

- `name_casefold`

- `name_sorted_tokens`

- `address_exact`

- `address_sorted_tokens`

- `address_house_signature`

- `name_address_combined`

- `country_name_casefold`


### Results

- Total ground-truth matches: **345,670**

- Retrieved ground-truth matches: **151,284**

- Pair-level blocking recall: **43.7654%**

- Entity-level complete recall: **11.0150%**

- Total candidate pairs: **1,231,347**

- Mean candidates per S1: **12.31**

- Median candidates per S1: **2.00**

- P95 candidates per S1: **70.00**

- Maximum candidates per S1: **460**

- Zero-candidate S1 entities: **13,090**

- Runtime: **1487.67 seconds**


### Block-level results


| block                   |   true_matches_retrieved |   total_true_matches |   pair_recall |   s1_entities_hit |   s1_entities_with_true_matches |   entity_hit_rate |
|:------------------------|-------------------------:|---------------------:|--------------:|------------------:|--------------------------------:|------------------:|
| name_exact              |                    16028 |               345670 |     0.0463679 |             13303 |                           94317 |         0.141046  |
| name_casefold           |                    37152 |               345670 |     0.107478  |             29481 |                           94317 |         0.312574  |
| name_sorted_tokens      |                    93014 |               345670 |     0.269083  |             59546 |                           94317 |         0.631339  |
| address_exact           |                     7671 |               345670 |     0.0221917 |              7667 |                           94317 |         0.0812897 |
| address_sorted_tokens   |                    40324 |               345670 |     0.116655  |             30508 |                           94317 |         0.323462  |
| address_house_signature |                    71279 |               345670 |     0.206205  |             43098 |                           94317 |         0.456948  |
| name_address_combined   |                     1925 |               345670 |     0.0055689 |              1855 |                           94317 |         0.0196677 |
| country_name_casefold   |                    37152 |               345670 |     0.107478  |             29481 |                           94317 |         0.312574  |


### Important interpretation


This experiment measures candidate-generation recall against the training ground truth. A blocking strategy is not considered sufficient merely because it produces a small candidate set; all true matches must remain available for the downstream pair model.


Business-name `digits` is intentionally not used as a standalone blocking key because M1 validation showed that its production recovery was only 1.4154% and that empty numeric representations must not be treated as matching evidence. It remains available as an auxiliary feature for later pair scoring.


The negative-collision analysis from M1 was sampled, so blocking collision behavior should also be examined empirically rather than interpreted as an exhaustive full-dataset collision probability.
