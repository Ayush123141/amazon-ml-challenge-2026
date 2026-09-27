# Score-maximization strategy

## Verified position

- The frozen 100K candidate set contains 1,799,071 pairs and 172,163 true links.
- It has 14,060 Source-1 entities with no candidate and candidate pair recall of about 49.7%.
- Its previously measured perfect-candidate oracle is Macro F0.5 **0.675934**.

Therefore, a pair scorer alone cannot reach the competition target. Its first purpose is to establish a high-precision baseline and measure how closely it approaches the current candidate oracle.

## Sequence of work

1. **Train and calibrate M3 models on the 100K artifact.** Use entity-disjoint train/validation splits and the canonical macro F0.5 metric, including zero-candidate entities. Start with logistic regression, then compare a tree-based model. Tune thresholds for macro F0.5 rather than assuming 0.5.
2. **Diagnose the gap to the 0.675934 oracle.** If the model is materially below it, improve features/model calibration first. If close, retrieval—not scoring—is proven to be the priority.
3. **Run only targeted retrieval experiments.** Keep the M2.2 candidates and union new candidates with them. Restrict every new method to same-country retrieval, cap top-K candidates per entity, and measure candidate oracle before fitting a model.
4. **Prioritize structured, high-recall additions for M2 misses.** The miss analysis already shows same-country pairs with high name/address similarity. Test normalized token-signatures (including legal-suffix removal), numeric-address + street signatures, and top-K character/token similarity retrieval. Reject any configuration that does not improve candidate oracle at a controlled candidate volume.
5. **Use a stronger pair model only after retrieval improves.** Evaluate a tree-based model on the same split and retain the model/threshold only when macro F0.5 improves.
6. **Scale only verified choices.** Rebuild full 2.2M candidates and generate a final submission only after the 100K candidate oracle and model score justify it.

## Required evidence for every experiment

- Candidate links/entity, zero-candidate entities, pair recall, and candidate-oracle macro F0.5.
- Pair precision/recall/F0.5 plus macro, singleton, and non-singleton F0.5.
- Entity-disjoint split seed, feature/model configuration, threshold, and artifact paths.
