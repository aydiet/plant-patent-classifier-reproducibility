# BERT baseline report (bert_baseline_20260129_094103)

- Created (UTC): 2026-01-29T08:57:10+00:00
- Git commit: ac11db0da270912f3954206595652aa7a08c6b9c
- Base model: anferico/bert-for-patents
- Device: mps
- Max length: 512 (truncation + padding=max_length)
- Precision target (val threshold selection): 0.9
- Chosen threshold: 0.994587
- Seed: 2026-01-29 (seed_int=1407623767)

## Data

Split counts (yes/no): train 142/278, val 30/60, test 31/59

## Token length audit

- Truncation rate (> max_length): 0.002
- Token lengths: p50=165, p90=261, p99=324

## Training args

```json
{
  "base_model": "anferico/bert-for-patents",
  "bf16": false,
  "created_at_utc": "2026-01-29T08:57:10+00:00",
  "device": "mps",
  "eval_batch_size": 2,
  "fp16": false,
  "git_commit": "ac11db0da270912f3954206595652aa7a08c6b9c",
  "gradient_accumulation_steps": 4,
  "gradient_checkpointing": true,
  "labels_csv": "metadata/labeling_candidates_600.csv",
  "learning_rate": 2e-05,
  "max_length": 512,
  "num_train_epochs": 5.0,
  "platform": "Darwin 25.2.0 (arm64)",
  "precision_target": 0.9,
  "python": "3.11.14",
  "run_id": "bert_baseline_20260129_094103",
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
  "accuracy": 0.9666666666666667,
  "f1": 0.9473684210526315,
  "pr_auc": 0.9898158051099227,
  "precision": 1.0,
  "recall": 0.9,
  "roc_auc": 0.9944444444444445
}
```

Confusion matrix (val):

| | pred=no | pred=yes |
|---|---:|---:|
| true=no | 60 | 0 |
| true=yes | 3 | 27 |


## Test (at chosen threshold)

```json
{
  "accuracy": 0.9444444444444444,
  "f1": 0.9180327868852459,
  "pr_auc": 0.9771802256288179,
  "precision": 0.9333333333333333,
  "recall": 0.9032258064516129,
  "roc_auc": 0.9885183160196829
}
```

Confusion matrix (test):

| | pred=no | pred=yes |
|---|---:|---:|
| true=no | 57 | 2 |
| true=yes | 3 | 28 |


## Outputs

- Model (HF-exportable): models/bert_for_patents_2026-01-29_safe/hf_export
- Checkpoints: models/bert_for_patents_2026-01-29_safe/checkpoints
- Val predictions: metadata/bert_for_patents_predictions_val_2026-01-29_safe.csv
- Test predictions: metadata/bert_for_patents_predictions_test_2026-01-29_safe.csv
