# Dataset Quality Leaderboard

> Updated 2026-08-21: v4 recovered from corruption (rebuilt from v3) and
> re-evaluated with geom_v2 mmwave features (134d Cartesian geometry +
> doppler/intensity stats). v5_structfeat re-extracted with the same
> features. mmwave probe val_acc: v4 0.34→0.77, v5_structfeat 0.51→0.76.
> See `results/quality_v4_after_recover.json` and
> `results/quality_v5_structfeat_after_fix.json`.

| dataset | InfoScore | CompactScore | CleanScore | Quality |
|---|---|---|---|---|
| datasets/mmfi/v1 | 0.103 | 0.287 | 1.000 | 0.356 |
| datasets/mmfi/v2 | 0.101 | 0.241 | 1.000 | 0.337 |
| datasets/mmfi/v4 | 0.267 | 0.848 | 1.000 | **0.646** |
| datasets/mmfi/v5_structfeat | 0.331 | 0.878 | 1.000 | **0.683** |
