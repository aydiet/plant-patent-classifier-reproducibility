# Variety vs technology report (variety_technology_20260130_152358)

- Created (UTC): 2026-01-30T14:30:32+00:00
- Git commit: 089d739f7f305693999aa939ac38dcce05f01f69
- Base model: mpi-inno-comp/paecter
- Device: mps
- Max length: 512 (truncation + padding=max_length)
- Threshold selection: max_f1
- Chosen threshold (technology): 0.000399
- Seed: 2026-01-30 (seed_int=2322758049)

## Data

Split counts (variety/technology): train 61/81, val 11/19, test 12/19

## Token length audit

- Truncation rate (> max_length): 0.000
- Token lengths: p50=144, p90=252, p99=319

## Training args

```json
{
  "base_model": "mpi-inno-comp/paecter",
  "bf16": false,
  "created_at_utc": "2026-01-30T14:30:32+00:00",
  "device": "mps",
  "eval_batch_size": 8,
  "fp16": false,
  "git_commit": "089d739f7f305693999aa939ac38dcce05f01f69",
  "gradient_accumulation_steps": 1,
  "gradient_checkpointing": false,
  "labels_csv": "metadata/labeling_candidates_600.csv",
  "learning_rate": 2e-05,
  "max_length": 512,
  "num_train_epochs": 8.0,
  "platform": "Darwin 25.2.0 (arm64)",
  "precision_target": null,
  "python": "3.11.14",
  "run_id": "variety_technology_20260130_152358",
  "seed_int": 2322758049,
  "seed_str": "2026-01-30",
  "splits_csv": "metadata/labeling_splits_420_90_90.csv",
  "torch": "2.10.0",
  "train_batch_size": 4,
  "transformers": "5.0.0",
  "warmup_ratio": 0.06,
  "weight_decay": 0.01
}
```

## Validation (at chosen threshold)

```json
{
  "accuracy": 0.9666666666666667,
  "f1": 0.9743589743589743,
  "pr_auc": 0.9739721590035257,
  "precision": 0.95,
  "recall": 1.0,
  "roc_auc": 0.9617224880382775
}
```

Confusion matrix (val):

| | pred=variety | pred=technology |
|---|---:|---:|
| true=variety | 10 | 1 |
| true=technology | 0 | 19 |


## Test (at chosen threshold)

```json
{
  "accuracy": 1.0,
  "f1": 1.0,
  "pr_auc": 0.9999999999999996,
  "precision": 1.0,
  "recall": 1.0,
  "roc_auc": 1.0
}
```

Confusion matrix (test):

| | pred=variety | pred=technology |
|---|---:|---:|
| true=variety | 12 | 0 |
| true=technology | 0 | 19 |


## Outputs

- Model (HF-exportable): models/variety_technology_paecter_2026-01-30/hf_export
- Checkpoints: models/variety_technology_paecter_2026-01-30/checkpoints
- Val predictions: metadata/variety_technology_paecter_predictions_val_2026-01-30.csv
- Test predictions: metadata/variety_technology_paecter_predictions_test_2026-01-30.csv
