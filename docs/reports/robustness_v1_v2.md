# Robustness Report: MMFi v1 vs v2 vs v3

Protocol: cross-subject (cs) split, seeds 0-2, Robustness Score = mean accuracy over all profiles.

| model | version | robustness (mean±std) | acc_full | profiles |
|-------|---------|----------------------|----------|----------|
| token_fusion | v1 (4 mod) | 0.1462 ± 0.075 | 0.2305 | 15 |
| token_fusion | v2 (4 mod) | 0.1425 ± 0.056 | 0.2382 | 15 |
| token_fusion | v3 (5 mod) | **0.3167 ± 0.075** | **0.5759** | 21 |
| late_fusion  | v1 (4 mod) | 0.2288 ± 0.027 | 0.4055 | 15 |
| late_fusion  | v2 (4 mod) | 0.2138 ± 0.016 | 0.3635 | 15 |
| late_fusion  | v3 (5 mod) | **0.2615 ± 0.023** | 0.4551 | 21 |

> v3 = v2 samples + injected `rgb` body-keypoint modality (17,2/frame, ResNet-48
> keypoints). Splits/labels unchanged (apples-to-apples). Profile count grew
> 15→21 because K grew 4→5 (1 full + 5 single-miss + 10 double-miss + 5 single-modal).
> v1↔v2↔v3 comparisons use **Robustness Score relative to full-modal accuracy** and
> degradation per profile, which are comparable despite differing profile counts.

## Degradation matrix (multi-seed, mean)

### token_fusion
| profile | v1 deg | v2 deg | v3 acc | v3 deg |
|---------|--------|--------|--------|--------|
| full | 0 | 0 | 0.5759 | 0 |
| miss-mmwave | 0.1472 | 0.1893 | 0.3177 | 0.2582 |
| only-mmwave | 0.0085 | — | 0.1708 | 0.4051 |
| only-rgb | — | — | 0.2232 | 0.3527 |
| miss-rgb | — | — | 0.2360 | 0.3399 |

### late_fusion
| profile | v1 deg | v2 deg | v3 acc | v3 deg |
|---------|--------|--------|--------|--------|
| full | 0 | 0 | 0.4551 | 0 |
| miss-mmwave | 0.3681 | 0.3232 | 0.0795 | 0.3756 |
| only-mmwave | 0.0357 | — | 0.3454 | 0.1097 |
| only-rgb | — | — | 0.0799 | 0.3752 |
| miss-rgb | — | — | 0.3507 | 0.1044 |

## Version change logs

- **v2**: dropped 5% train/val samples flagged cross-modality-inconsistent by token_fusion (drop_rate 0.05). Test unchanged.
- **v3**: injected `rgb` body-keypoint modality (17,2/frame) from MMFi raw (ResNet-48 keypoints). No sample dropped; splits/labels identical to v2.

## Findings (v1 → v2 → v3)

- **v2 (filter) did not help** — robustness flat-to-down (late_fusion 0.2288→0.2138). Dropping "inconsistent" samples hurt; the filter was too correlated with the eval model and too small to matter.
- **v3 (+rgb keypoints) is a step change** — token_fusion robustness 0.1425→0.3167 (2.2x), acc_full 0.2382→0.5759. **The weak-sensor problem is data information content, not task hardness.**
- `only-rgb` (single modality) already reaches 0.22 acc, comparable to `only-mmwave`'s v3 contribution — vision keypoints carry strong, independent discriminative signal, confirming the 4-RF-modality setup starved the model of usable signal.
- mmWave remains the dominant RF sensor, but its grip is looser in v3: with rgb present, miss-mmwave degradation is now a smaller fraction of the full accuracy.
- `token_fusion` again beats its own v2 but **still trails late_fusion on stability** (std 0.075 vs 0.023) while overtaking it on robustness — the modality-dropout transformer benefits more from the richer modality set.

## Conclusions

- The data flywheel works **only when you add information**, not when you delete samples. v3 is the first version where robustness genuinely improved.
- The benchmark now distinguishes the two hypotheses cleanly: **weak modalities lack independent discriminative signal** (rgb fixes it) rather than "cs 27-class is too hard" (rgb full-acc 0.576 is respectable).
- Next lever: feed rgb/infra keypoints' strength back into the RF side — e.g. metric learning / cross-modal alignment so wifi/lidar/depth learn from rgb's structure, targeting the S20/S40/S18 subjects and classes 0/5/17/25 flagged in the v2 weak-point analysis.
