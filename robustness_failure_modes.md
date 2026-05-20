# Robustness Failure-Mode Ledger

## FM-1: Underpowered sensitivity strata
- **Trigger condition:** Slice/domain/query partitions with `n_pairs < 20`.
- **Observed evidence:** canonical v2 buckets are marked exploratory (`n=0..3` by slice); per-domain counts are 2/2/2.
- **Effect on robustness interpretation:** inferential conclusions are null for confirmatory claims in those strata.

## FM-2: Latency practical-threshold violation
- **Trigger condition:** Kalman/Mean latency ratio exceeds canonical bound (`>1.5`).
- **Observed evidence:** ratio `2.0554` in canonical v2.
- **Effect on robustness interpretation:** practical-significance criterion fails even when raw quality delta is positive.

## FM-3: Calibration-to-retrieval transfer null
- **Trigger condition:** Calibration objective changes but downstream Kalman-vs-Mean retrieval delta does not move.
- **Observed evidence:** downstream delta change `0.0` on validation and `0.0` on test under powered calibration selection.
- **Effect on robustness interpretation:** robustness to calibration choice remains null in downstream retrieval metric space.

## FM-4: Missing sensitivity axes in committed artifacts
- **Trigger condition:** no multi-seed, no query-length sweep, no specialist-count sweep in cited artifacts.
- **Observed evidence:** single canonical seed (`42`); no committed query-length or specialist-count contrast table.
- **Effect on robustness interpretation:** these sensitivity dimensions remain unestimated (null evidence state).

## FM-5: Confidence-interval inflation under low sample count
- **Trigger condition:** very small paired test set.
- **Observed evidence:** canonical v2 nDCG@10 CI width `0.2847` (`n_pairs=6`) vs confirmatory reference width `0.002534` (`n_pairs=1193`).
- **Effect on robustness interpretation:** large interval widening increases decision instability and weakens robustness claims.
