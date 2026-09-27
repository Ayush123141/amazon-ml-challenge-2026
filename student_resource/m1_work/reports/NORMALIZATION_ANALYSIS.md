# Normalization Analysis Report

Amazon ML Challenge 2026 — Business Entity Resolution

## 1. Purpose

This analysis evaluates candidate normalization transformations using sampled positive matched pairs and sampled negative non-matching pairs.

The objective is to identify transformations that recover true variations while measuring the observed collision risk on non-matching pairs.

## 2. Dataset used

- Positive pairs analyzed: 500,000
- Negative pairs analyzed: 1,000,000
- Random seed: 42

## 3. Candidate transformations

The following representations were evaluated:

- `original`
- `unicode`
- `casefold`
- `whitespace`
- `punctuation`
- `alphanumeric`
- `sorted_tokens`
- `digits`

## 4. Business-name recovery

| Transformation | Recovery | Incremental Recovery | Negative Collision |
|---|---:|---:|---:|
| `original` | 4.65% | 4.65% | 0.00% |
| `unicode` | 4.65% | 0.00% | 0.00% |
| `casefold` | 10.74% | 6.09% | 0.00% |
| `whitespace` | 15.73% | 5.00% | 0.00% |
| `punctuation` | 21.65% | 5.91% | 0.00% |
| `alphanumeric` | 22.92% | 1.28% | 0.00% |
| `sorted_tokens` | 26.68% | 3.75% | 0.00% |
| `digits` | 95.86% | 69.18% | 0.00% |

## 5. Business-address recovery

| Transformation | Recovery | Incremental Recovery | Negative Collision |
|---|---:|---:|---:|
| `original` | 2.22% | 2.22% | 0.00% |
| `unicode` | 2.22% | 0.00% | 0.00% |
| `casefold` | 7.24% | 5.02% | 0.00% |
| `whitespace` | 7.41% | 0.17% | 0.00% |
| `punctuation` | 8.28% | 0.87% | 0.00% |
| `alphanumeric` | 8.33% | 0.05% | 0.00% |
| `sorted_tokens` | 11.70% | 3.37% | 0.00% |
| `digits` | 62.03% | 50.32% | 0.04% |

## 6. Interpretation

The recovery percentage measures how many sampled positive pairs become exactly equal after a transformation.

The negative collision percentage measures how many sampled non-matching pairs become exactly equal after a transformation.

A transformation should not be selected only because it increases positive recovery. Collision behavior must also be considered because false merges are costly for this task.

## 7. Important implementation principle

Normalization should generate multiple representations rather than replacing the original value.

Recommended conceptual structure:

Original → Unicode → Casefold → Punctuation → Alphanumeric → Token → Sorted Tokens → Digits

The original representation should remain available for final matching and explainability.

## 8. Negative-pair analysis limitations

Negative pairs were sampled rather than exhaustively enumerated because the complete cross-source Cartesian product is extremely large.

Therefore, observed collision rates describe the sampled negative population and should not be interpreted as an exact full-dataset collision probability.

## 9. Files generated

- `positive_name_recovery.csv`
- `positive_address_recovery.csv`
- `negative_name_collision.csv`
- `negative_address_collision.csv`
- `country_name_recovery.csv`
- `country_address_recovery.csv`
- `transformation_examples.csv`
- `transformation_comparison.csv`
- `normalization_recommendations.csv`

## 10. Next stage

Use the measured recovery and collision results to design the production `text_normalizer.py` module.

The production implementation should preserve multiple representations and avoid aggressive transformations that can collapse distinct businesses.
