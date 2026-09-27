# M3 implementation plan

1. **Feature validation (current)** — Expand the frozen 100K M2 candidate rows, attach true labels from ground truth, and produce reproducible pair features. Validate a small smoke artifact before creating the 100K artifact.
2. **Entity-level split** — Split by `source1_entity_id` (80/20, fixed seed) so no entity's candidate pairs cross train and validation.
3. **Baseline matcher** — Train a class-weighted logistic-regression model on the engineered numeric features and save predictions/configuration.
4. **Metric-first threshold selection** — Convert pair probabilities to per-entity prediction sets; optimize a threshold grid using the canonical macro F0.5 evaluator, reporting singleton and non-singleton metrics.
5. **Stronger classical model** — Use the available gradient-boosting library if installed; otherwise use sklearn histogram gradient boosting. Compare only on the same entity split.
6. **Decision gate** — Compare the best validation score to the M2 candidate oracle (0.675934). If it is close, run one justified retrieval experiment; if materially below, improve M3 features/model first.

Guardrails: M2 remains frozen, all experiments retain singleton empty-prediction behavior, and no full 2.2M candidate generation occurs before this 100K pipeline is validated.
