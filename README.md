# SensorBench

A data/model-decoupled benchmark framework for **cross-modal sensor fusion with missing-modality robustness**, applied to indoor human activity sensing.

The **dataset is the product**; the model is a measuring stick. Data quality is quantified by *robustness*: train on all sensors, evaluate how accuracy degrades when sensors go missing. Better data = smaller degradation.

## Core design

1. **Dataset (model-agnostic contract).** Canonical per-sample format (`framework/dataset/sample.py`): `{id, label, modalities: {sensor: Modality(data, frame_indices, sample_rate)}, text, meta}`. Splits, modality registry (`modalities.yaml`), and versioning (`meta.json` + changelog) all live on the dataset side.
2. **Models are pluggable.** Anything implementing `SensorModel` (`framework/models/base.py`: `fit` + `predict(sample, available)`) can be swapped in. Missing-modality behavior is the model's own concern.
3. **Evaluation protocol is fixed and reproducible.** `protocol.json` auto-generates 15 modality profiles (full / 4 single-missing / 6 double-missing / 4 single-modal). **Robustness Score** = mean accuracy over all profiles, seeds 0-2, mean ± std.
4. **Data flywheel.** v1 → evaluate → find weak (modality, class, subject) → clean/enrich → v2 → robustness should improve.

## Layout

```
framework/dataset     # sample contract, loader, subject-stratified splits
framework/models      # SensorModel protocol, token_fusion (main), late_fusion (baseline)
framework/harness     # protocol builder, evaluator, leaderboard
curation/ingest       # MMFi raw -> canonical samples
curation/clean        # frame-alignment check, cross-modal consistency filter
curation/version      # meta.json / changelog
scripts/              # ingest_mmfi.py, train.py, run_eval.py, make_v2.py
datasets/mmfi/v1      # generated (not committed)
```

## Quickstart

Environment: use the `sensorbench` conda env (torch 2.9.1+cu128, scipy, opencv, numpy, pyyaml, pytest, transformers, streamlit, plotly).

```bash
conda activate sensorbench
```

```bash
# 1. Ingest MMFi (cs split; 4 modalities wifi/mmwave/lidar/depth, 27 classes)
python scripts/ingest_mmfi.py \
  --annotations-train .../mmfi_train_cs_full.json \
  --annotations-test  .../mmfi_test_cs_full.json \
  --raw-root .../MMFi_Dataset --out datasets/mmfi/v1

# 2. Generate the 15-profile protocol
python -c "import json; from framework.harness.protocol import build_protocol; \
  json.dump(build_protocol(['wifi','depth','lidar','mmwave'], seeds=[0,1,2]), open('protocol.json','w'))"

# 3. Train a model (one seed per invocation; each loads the full dataset -> run sequentially)
python scripts/train.py --dataset datasets/mmfi/v1 --model token_fusion --seed 0 --epochs 30

# 4. Evaluate all seeds and build the leaderboard
python scripts/run_eval.py --dataset datasets/mmfi/v1 --protocol protocol.json \
  --ckpt-dir checkpoints --seeds 0,1,2 --out leaderboard_v1.json
```

## Adding a model

Implement `SensorModel` (see `framework/models/token_fusion.py` for the reference):
- `fit(train, val, cfg)` — training loop
- `predict(sample, available) -> {class_id: prob}` — missing modalities are the caller's decision

Register it in `scripts/train.py` + `scripts/run_eval.py` `MODELS` dict.

## Adding a dataset

Write an ingest adapter under `curation/ingest/` that produces canonical samples (see `mmfi.py`), then reuse the same loader/train/eval pipeline unchanged. New sensor = register it in `modalities.yaml`.

## Results

See `docs/reports/robustness_v1_v2.md` for the v1/v2 robustness comparison.
