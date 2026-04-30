# BERT baseline report (bert_baseline_20260129_100306)

- Created (UTC): 2026-01-29T09:06:49+00:00
- Git commit: ac11db0da270912f3954206595652aa7a08c6b9c
- Base model: bert-base-uncased
- Device: mps
- Max length: 512 (truncation + padding=max_length)
- Precision target (val threshold selection): 0.9
- Chosen threshold: 0.038927
- Seed: 2026-01-28 (seed_int=3475473504)

## Data

Split counts (yes/no): train 142/278, val 30/60, test 31/59

## Token length audit

- Truncation rate (> max_length): 0.003
- Token lengths: p50=183, p90=290, p99=354

## Training args

```json
{
  "base_model": "bert-base-uncased",
  "bf16": false,
  "created_at_utc": "2026-01-29T09:06:49+00:00",
  "device": "mps",
  "eval_batch_size": 16,
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
  "run_id": "bert_baseline_20260129_100306",
  "seed_int": 3475473504,
  "seed_str": "2026-01-28",
  "splits_csv": "metadata/labeling_splits_420_90_90.csv",
  "torch": "2.10.0",
  "train_batch_size": 8,
  "transformers": "5.0.0",
  "warmup_ratio": 0.06,
  "weight_decay": 0.01
}
```

## Validation (at chosen threshold)

```json
{
  "accuracy": 0.9555555555555556,
  "f1": 0.9354838709677419,
  "pr_auc": 0.9781910233046244,
  "precision": 0.90625,
  "recall": 0.9666666666666667,
  "roc_auc": 0.9894444444444445
}
```

Confusion matrix (val):

| | pred=no | pred=yes |
|---|---:|---:|
| true=no | 57 | 3 |
| true=yes | 1 | 29 |


## Test (at chosen threshold)

```json
{
  "accuracy": 0.9555555555555556,
  "f1": 0.9393939393939394,
  "pr_auc": 0.9769726385056537,
  "precision": 0.8857142857142857,
  "recall": 1.0,
  "roc_auc": 0.9879715691634773
}
```

Confusion matrix (test):

| | pred=no | pred=yes |
|---|---:|---:|
| true=no | 55 | 4 |
| true=yes | 0 | 31 |


## Outputs

- Model (HF-exportable): models/bert_base_uncased_2026-01-28/hf_export
- Checkpoints: models/bert_base_uncased_2026-01-28/checkpoints
- Val predictions: metadata/bert_base_uncased_predictions_val_2026-01-28.csv
- Test predictions: metadata/bert_base_uncased_predictions_test_2026-01-28.csv
