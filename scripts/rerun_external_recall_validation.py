#!/usr/bin/env python
"""
Re-evaluate first-stage plant-related classifiers on the external recall set.

The external recall set is positive-only. This script therefore reports recall
and score distributions, not precision, calibration, F1, ROC-AUC, or PR-AUC.
It can evaluate both the validation-selected thresholds used in the manuscript
and a fixed probability threshold of 0.5.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_ORDER = ["bert-base-uncased", "bert-for-patents", "PaECTER"]
VALIDATION_THRESHOLDS = {
    "bert-base-uncased": 0.038927,
    "bert-for-patents": 0.994587,
    "PaECTER": 0.992003,
}
CHECKPOINTS = {
    "bert-base-uncased": "models/bert_base_uncased_2026-01-28/hf_export",
    "bert-for-patents": "models/bert_for_patents_2026-01-29_safe/hf_export",
    "PaECTER": "models/paecter_2026-01-29_rerun/hf_export",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        default="metadata/gold_test_external_combined_positives.parquet",
        help="Positive-only external recall parquet.",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "validation-thresholds", "fixed050"],
        default="both",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def load_external_text(base: Path, parquet_path: str) -> pd.DataFrame:
    gold_df = pd.read_parquet(base / parquet_path)
    has_title = gold_df["appln_title_en_final"].notna() & (
        gold_df["appln_title_en_final"].str.len() > 0
    )
    has_abstract = gold_df["appln_abstract_en_final"].notna() & (
        gold_df["appln_abstract_en_final"].str.len() > 0
    )
    full_text = gold_df[has_title & has_abstract].copy()
    full_text["input_text"] = (
        full_text["appln_title_en_final"] + "\n\n" + full_text["appln_abstract_en_final"]
    )
    if "gold_source" not in full_text.columns:
        full_text["gold_source"] = "external"
    return full_text


def score_model(
    *,
    base: Path,
    model_name: str,
    texts: list[str],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(str(base / CHECKPOINTS[model_name]))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(base / CHECKPOINTS[model_name])
    ).to(device)
    model.eval()

    probs: list[float] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            logits = model(**inputs).logits
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

    del model, tokenizer
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return np.array(probs)


def write_results(
    *,
    base: Path,
    full_text: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    tag: str,
) -> None:
    rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []

    for model_name in MODEL_ORDER:
        probs = probabilities[model_name]
        threshold = thresholds[model_name]
        preds = (probs >= threshold).astype(int)
        true_positives = int(preds.sum())
        false_negatives = int(len(preds) - true_positives)

        rows.append(
            {
                "model": model_name,
                "n_positives": len(preds),
                "threshold": threshold,
                "recall": true_positives / len(preds),
                "true_positives": true_positives,
                "false_negatives": false_negatives,
                "mean_probability": float(probs.mean()),
                "median_probability": float(np.median(probs)),
                "min_probability": float(probs.min()),
            }
        )

        for idx, (_, row) in enumerate(full_text.iterrows()):
            pred_rows.append(
                {
                    "model": model_name,
                    "docdb_family_id": row["docdb_family_id"],
                    "gold_source": row["gold_source"],
                    "probability": float(probs[idx]),
                    "predicted": int(preds[idx]),
                    "gold_label": 1,
                }
            )

    pd.DataFrame(rows).to_csv(base / f"metadata/external_recall_{tag}_results_2026-03-03.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(
        base / f"metadata/external_recall_{tag}_predictions_2026-03-03.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parents[1]
    full_text = load_external_text(base, args.parquet)
    texts = full_text["input_text"].tolist()
    print(f"Evaluating {len(full_text)} positive external families")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    probabilities: dict[str, np.ndarray] = {}
    for model_name in MODEL_ORDER:
        print(f"Scoring {model_name}...")
        probabilities[model_name] = score_model(
            base=base,
            model_name=model_name,
            texts=texts,
            batch_size=args.batch_size,
            device=device,
        )

    if args.mode in {"both", "validation-thresholds"}:
        write_results(
            base=base,
            full_text=full_text,
            probabilities=probabilities,
            thresholds=VALIDATION_THRESHOLDS,
            tag="pos_only",
        )
        print("Saved validation-threshold external recall results.")

    if args.mode in {"both", "fixed050"}:
        write_results(
            base=base,
            full_text=full_text,
            probabilities=probabilities,
            thresholds={model_name: 0.5 for model_name in MODEL_ORDER},
            tag="fixed050",
        )
        print("Saved fixed-threshold external recall results.")


if __name__ == "__main__":
    main()
