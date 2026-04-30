# Model comparison (2026-01-29)

All results below use the same labeled dataset and deterministic split manifest:

- Labels: `metadata/labeling_candidates_600.csv`
- Splits: `metadata/labeling_splits_420_90_90.csv`
- Unit of analysis: `docdb_family_id`
- Text field: `title_en + "\n\n" + abstract_en`
- Max length: 512 (truncation + padding=max_length)
- Threshold selection: pick a validation-set decision threshold to satisfy `precision >= 0.90`.

## Summary table

| Model | Report | Chosen threshold | Val (PR-AUC / Precision / Recall / F1) | Test (PR-AUC / Precision / Recall / F1) |
|---|---|---:|---|---|
| `bert-base-uncased` | `metadata/bert_base_uncased_report_2026-01-28.md` | 0.038927 | 0.9782 / 0.9063 / 0.9667 / 0.9355 | 0.9770 / 0.8857 / 1.0000 / 0.9394 |
| `anferico/bert-for-patents` (safe MPS run) | `metadata/bert_for_patents_report_2026-01-29_safe.md` | 0.994587 | 0.9898 / 1.0000 / 0.9000 / 0.9474 | 0.9772 / 0.9333 / 0.9032 / 0.9180 |
| `mpi-inno-comp/paecter` (rerun) | `metadata/paecter_report_2026-01-29_rerun.md` | 0.992003 | 0.9989 / 0.9677 / 1.0000 / 0.9836 | 0.9761 / 0.9091 / 0.9677 / 0.9375 |

## Takeaways

- These three models are **very close on test PR-AUC** (≈ 0.976–0.977).
- The most meaningful difference is the **operating point** (false positives vs false negatives), which is driven by the chosen threshold.
- If optimizing for **fewer false positives** (precision leaning): `bert-for-patents` (safe) has the best test precision among the three.
- If optimizing for **fewer false negatives** (recall leaning): `bert-base-uncased` hit perfect test recall in this run.
- `paecter` is consistently strong and stable; the rerun confirms no MPS OOM issues under conservative settings.

## Notes on the earlier unstable `bert-for-patents` run

An earlier `anferico/bert-for-patents` run (see `metadata/bert_for_patents_report_2026-01-28.md`) showed MPS out-of-memory warnings and a collapsed classifier (predicting “yes” for almost everything). The safe rerun eliminates the MPS issues and produces strong metrics.
