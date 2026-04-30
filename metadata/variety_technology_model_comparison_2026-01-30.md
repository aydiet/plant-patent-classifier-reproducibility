# Variety vs technology: model comparison (2026-01-30T14:30:33+00:00)

Models:
- bert-base-uncased
- anferico/bert-for-patents
- mpi-inno-comp/paecter

## At each model's chosen threshold

| model | threshold_method | threshold(tech) | val_f1 | val_precision | val_recall | test_f1 | test_precision | test_recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bert-base-uncased | max_f1 | 0.005037 | 0.9500 | 0.9048 | 1.0000 | 0.8837 | 0.7917 | 1.0000 |
| anferico/bert-for-patents | max_f1 | 0.000114 | 0.9744 | 0.9500 | 1.0000 | 0.9744 | 0.9500 | 1.0000 |
| mpi-inno-comp/paecter | max_f1 | 0.000399 | 0.9744 | 0.9500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Fixed threshold: 0.5

| model | val_f1 | val_precision | val_recall | test_f1 | test_precision | test_recall |
|---|---:|---:|---:|---:|---:|---:|
| bert-base-uncased | 0.9744 | 0.9500 | 1.0000 | 0.9744 | 0.9500 | 1.0000 |
| anferico/bert-for-patents | 0.9474 | 0.9474 | 0.9474 | 0.9744 | 0.9500 | 1.0000 |
| mpi-inno-comp/paecter | 0.9189 | 0.9444 | 0.8947 | 1.0000 | 1.0000 | 1.0000 |

## Fixed threshold: 0.9

| model | val_f1 | val_precision | val_recall | test_f1 | test_precision | test_recall |
|---|---:|---:|---:|---:|---:|---:|
| bert-base-uncased | 0.9189 | 0.9444 | 0.8947 | 1.0000 | 1.0000 | 1.0000 |
| anferico/bert-for-patents | 0.9474 | 0.9474 | 0.9474 | 0.9744 | 0.9500 | 1.0000 |
| mpi-inno-comp/paecter | 0.9189 | 0.9444 | 0.8947 | 1.0000 | 1.0000 | 1.0000 |

## Artifacts

- bert-base-uncased report: metadata/variety_technology_bert_base_uncased_report_2026-01-30.md
- bert-base-uncased metrics: metadata/variety_technology_bert_base_uncased_metrics_2026-01-30.json
- anferico/bert-for-patents report: metadata/variety_technology_bert_for_patents_report_2026-01-30.md
- anferico/bert-for-patents metrics: metadata/variety_technology_bert_for_patents_metrics_2026-01-30.json
- mpi-inno-comp/paecter report: metadata/variety_technology_paecter_report_2026-01-30.md
- mpi-inno-comp/paecter metrics: metadata/variety_technology_paecter_metrics_2026-01-30.json
