# Evidence review: Data Engineering & Decision Optimisation

Independent portfolio research. Updated 2026-10-09. Development and documentation include AI assistance; the committed executable code, data provenance and test outputs establish the work products. No institutional endorsement or third-party authorship review is claimed.

## Research purpose

Large-scale data processing and constrained optimisation.

## Published evidence

24.08m raw TLC trips; complete zero-inclusive zone panel; 100 offline transport cases including 40 exhaustive oracle comparisons.

## Interpretation boundary

TLC trip flow does not observe idle fleet supply; centroid distance is a proxy, and allocation is a static model.

## Review standard

Check the source data and split before interpreting a score. Compare the strongest result with a simple baseline. Inspect failed diagnostics and uncertainty. Reproduce the calculations from the documented environment; distinguish measured findings from simulated or assumed scenarios. The tests cover specific documented invariants and do not establish complete production correctness.

## Next research extension

Separate training and later evaluation periods, retain the aggregate input panel and stress uncertain supply/demand. Validate a decision objective using real availability/routing evidence before claiming operational savings.
