# Real-Data Normalizer Validation

Amazon ML Challenge 2026 — Business Entity Resolution

## 1. Validation scope

- Positive pairs analyzed: 500,000
- Negative pairs analyzed: 1,000,000
- Random seed: 42

The production `text_normalizer.py` was evaluated against real training positive pairs and sampled negative pairs.

## 2. Business-name validation

| Representation | Recovery | Negative Collision |
|---|---:|---:|
| `original` | 4.6492% | 0.0001% |
| `unicode` | 4.6492% | 0.0001% |
| `casefold` | 10.7378% | 0.0001% |
| `whitespace` | 15.7344% | 0.0001% |
| `punctuation` | 21.6466% | 0.0001% |
| `alphanumeric` | 22.9230% | 0.0001% |
| `sorted_tokens` | 26.6774% | 0.0001% |
| `digits` | 1.4154% | 0.0015% |

## 3. Business-address validation

| Representation | Recovery | Negative Collision |
|---|---:|---:|
| `original` | 2.2220% | 0.0000% |
| `unicode` | 2.2220% | 0.0000% |
| `casefold` | 7.2428% | 0.0000% |
| `whitespace` | 7.4124% | 0.0000% |
| `punctuation` | 8.2838% | 0.0000% |
| `alphanumeric` | 8.3298% | 0.0000% |
| `sorted_tokens` | 11.7040% | 0.0000% |
| `digits` | 59.1352% | 0.0435% |

## 4. Task 4 reproducibility

Task 6 recovery results were compared with the Task 4 normalization-analysis baseline.

| Field | Representation | Task 4 | Task 6 | Difference |
|---|---|---:|---:|---:|
| business_name | `original` | 4.6500% | 4.6492% | -0.0008% |
| business_name | `unicode` | 4.6500% | 4.6492% | -0.0008% |
| business_name | `casefold` | 10.7400% | 10.7378% | -0.0022% |
| business_name | `whitespace` | 15.7300% | 15.7344% | +0.0044% |
| business_name | `punctuation` | 21.6500% | 21.6466% | -0.0034% |
| business_name | `alphanumeric` | 22.9200% | 22.9230% | +0.0030% |
| business_name | `sorted_tokens` | 26.6800% | 26.6774% | -0.0026% |
| business_name | `digits` | 95.8600% | 1.4154% | -94.4446% |
| business_address | `original` | 2.2200% | 2.2220% | +0.0020% |
| business_address | `unicode` | 2.2200% | 2.2220% | +0.0020% |
| business_address | `casefold` | 7.2400% | 7.2428% | +0.0028% |
| business_address | `whitespace` | 7.4100% | 7.4124% | +0.0024% |
| business_address | `punctuation` | 8.2800% | 8.2838% | +0.0038% |
| business_address | `alphanumeric` | 8.3300% | 8.3298% | -0.0002% |
| business_address | `sorted_tokens` | 11.7000% | 11.7040% | +0.0040% |
| business_address | `digits` | 62.0300% | 59.1352% | -2.8948% |

## 5. Interpretation

The production normalizer is considered consistent with the Task 4 analysis when recovery values are close to the corresponding sampled baseline.

Negative collision results describe the sampled negative population and are not an exhaustive estimate of all possible cross-source collisions.

Representations should be used as separate signals rather than treating every normalized form as a standalone identity key.

## 6. Additional validation

Country-level and source-level validation tables were generated separately as CSV outputs.

## 7. Output files

- `address_collision_examples.csv`
- `country_address_validation.csv`
- `country_name_validation.csv`
- `name_collision_examples.csv`
- `real_negative_address_validation.csv`
- `real_negative_name_validation.csv`
- `real_normalizer_metrics.csv`
- `real_positive_address_validation.csv`
- `real_positive_name_validation.csv`
- `source_address_validation.csv`
- `source_name_validation.csv`
- `task4_vs_task6_comparison.csv`

## 8. Status

Real-data normalizer validation completed. Results should be reviewed before exposing representations to the blocking/retrieval stage.