# BERT baseline report (bert_baseline_20260129_101114)

- Created (UTC): 2026-01-29T09:24:01+00:00
- Git commit: ac11db0da270912f3954206595652aa7a08c6b9c
- Base model: mpi-inno-comp/paecter
- Device: mps
- Max length: 512 (truncation + padding=max_length)
- Precision target (val threshold selection): 0.9
- Chosen threshold: 0.992003
- Seed: 2026-01-29 (seed_int=1407623767)

## Data

Split counts (yes/no): train 142/278, val 30/60, test 31/59

## Token length audit

- Truncation rate (> max_length): 0.002
- Token lengths: p50=172, p90=270, p99=330

## Training args

```json
{
  "base_model": "mpi-inno-comp/paecter",
  "bf16": false,
  "created_at_utc": "2026-01-29T09:24:01+00:00",
  "device": "mps",
  "eval_batch_size": 2,
  "fp16": false,
  "git_commit": "ac11db0da270912f3954206595652aa7a08c6b9c",
  "gradient_accumulation_steps": 1,
  "gradient_checkpointing": false,
  "labels_csv": "metadata/labeling_candidates_600.csv",
  "learning_rate": 2e-05,
  "max_length": 512,
  "num_train_epochs": 5.0,
  "platform": "Darwin 25.2.0 (arm64)",
  "precision_target": 0.9,
  "python": "3.11.14",
  "run_id": "bert_baseline_20260129_101114",
  "seed_int": 1407623767,
  "seed_str": "2026-01-29",
  "splits_csv": "metadata/labeling_splits_420_90_90.csv",
  "torch": "2.10.0",
  "train_batch_size": 2,
  "transformers": "5.0.0",
  "warmup_ratio": 0.06,
  "weight_decay": 0.01
}
```

## Validation (at chosen threshold)

```json
{
  "accuracy": 0.9888888888888889,
  "f1": 0.9836065573770492,
  "pr_auc": 0.9989247311827957,
  "precision": 0.967741935483871,
  "recall": 1.0,
  "roc_auc": 0.9994444444444444
}
```

Confusion matrix (val):

| | pred=no | pred=yes |
|---|---:|---:|
| true=no | 59 | 1 |
| true=yes | 0 | 30 |


## Test (at chosen threshold)

```json
{
  "accuracy": 0.9555555555555556,
  "f1": 0.9375,
  "pr_auc": 0.9760948445728314,
  "precision": 0.9090909090909091,
  "recall": 0.967741935483871,
  "roc_auc": 0.9879715691634774
}
```

Confusion matrix (test):

| | pred=no | pred=yes |
|---|---:|---:|
| true=no | 56 | 3 |
| true=yes | 1 | 30 |


## Outputs

- Model (HF-exportable): models/paecter_2026-01-29_rerun/hf_export
- Checkpoints: models/paecter_2026-01-29_rerun/checkpoints
- Val predictions: metadata/paecter_predictions_val_2026-01-29_rerun.csv
- Test predictions: metadata/paecter_predictions_test_2026-01-29_rerun.csv
