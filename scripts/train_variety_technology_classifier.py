#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import random
import subprocess
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# HF Hub defaults to short (10s) timeouts; bump them for robustness.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


LABEL2ID = {"variety": 0, "technology": 1}
ID2LABEL = {0: "variety", 1: "technology"}


@dataclass(frozen=True)
class Example:
    docdb_family_id: int
    text: str
    y: int
    split: str


class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        encodings: dict[str, Any],
        labels: list[int],
    ) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return out or None
    except Exception:
        return None


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stable_seed_int(seed_str: str) -> int:
    # Stable across machines.
    h = hashlib.md5(seed_str.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _read_split_manifest(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        required = {"docdb_family_id", "split"}
        missing = required - set(r.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing columns in {path}: {sorted(missing)}")

        out: dict[int, str] = {}
        for i, row in enumerate(r, start=2):
            fam = (row.get("docdb_family_id") or "").strip()
            split = (row.get("split") or "").strip().lower()
            if not fam:
                raise SystemExit(f"Empty docdb_family_id at {path} line {i}")
            if split not in {"train", "val", "test"}:
                raise SystemExit(
                    f"Unexpected split={split!r} at {path} line {i} (expected train/val/test)"
                )
            fam_id = int(fam)
            if fam_id in out:
                raise SystemExit(f"Duplicate docdb_family_id {fam_id} in {path}")
            out[fam_id] = split
        return out


def _read_variety_technology_examples(
    labels_csv: Path, split_by_family: dict[int, str]
) -> list[Example]:
    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        required = {"docdb_family_id", "label", "subtype"}
        missing = required - set(r.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing columns in {labels_csv}: {sorted(missing)}")

        examples: list[Example] = []
        for i, row in enumerate(r, start=2):
            raw_id = (row.get("docdb_family_id") or "").strip()
            if not raw_id:
                raise SystemExit(f"Empty docdb_family_id at {labels_csv} line {i}")
            fam_id = int(raw_id)

            raw_label = (row.get("label") or "").strip().lower()
            if raw_label != "yes":
                continue

            raw_subtype = (row.get("subtype") or "").strip().lower()
            if raw_subtype not in LABEL2ID:
                continue

            # Prefer explicit text_en if present; else fall back to title+abstract.
            text_en = (row.get("text_en") or "").strip()
            if text_en:
                text = text_en
            else:
                title = (row.get("title_en") or "").strip()
                abstract = (row.get("abstract_en") or "").strip()
                text = (title + "\n\n" + abstract).strip()

            if not text:
                raise SystemExit(f"Empty text for family {fam_id} at {labels_csv} line {i}")

            split = split_by_family.get(fam_id)
            if split is None:
                raise SystemExit(
                    f"Family {fam_id} in {labels_csv} has no split assignment in split manifest"
                )

            examples.append(
                Example(
                    docdb_family_id=fam_id,
                    text=text,
                    y=LABEL2ID[raw_subtype],
                    split=split,
                )
            )

    fams = [e.docdb_family_id for e in examples]
    if len(fams) != len(set(fams)):
        raise SystemExit(f"Duplicate docdb_family_id values found in {labels_csv}")

    return examples


def _split_counts(examples: Iterable[Example]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {
        "train": {"variety": 0, "technology": 0},
        "val": {"variety": 0, "technology": 0},
        "test": {"variety": 0, "technology": 0},
    }
    for e in examples:
        out[e.split][ID2LABEL[e.y]] += 1
    return out


def _slugify_model_id(model_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", model_id.strip())
    slug = slug.strip("_")
    return slug or "model"


def _device_summary() -> str:
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.get_device_name(0)}"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _is_mps() -> bool:
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def _empty_device_cache() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if _is_mps() and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    except Exception:
        return


class _DeviceCacheCleanupCallback(TrainerCallback):
    def __init__(self, *, enabled: bool) -> None:
        super().__init__()
        self._enabled = enabled

    def _cleanup(self) -> None:
        if not self._enabled:
            return
        gc.collect()
        _empty_device_cache()

    def on_evaluate(self, args, state, control, **kwargs):  # type: ignore[override]
        self._cleanup()
        return control

    def on_save(self, args, state, control, **kwargs):  # type: ignore[override]
        self._cleanup()
        return control

    def on_train_end(self, args, state, control, **kwargs):  # type: ignore[override]
        self._cleanup()
        return control


def _compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    labels = np.asarray(labels)
    logits = np.asarray(logits)

    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    p_tech = probs[:, 1]
    y_pred = (p_tech >= 0.5).astype(int)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(labels, y_pred)),
        "precision": float(precision_score(labels, y_pred, zero_division=0)),
        "recall": float(recall_score(labels, y_pred, zero_division=0)),
        "f1": float(f1_score(labels, y_pred, zero_division=0)),
    }

    if len(set(labels.tolist())) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, p_tech))
        metrics["pr_auc"] = float(average_precision_score(labels, p_tech))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    return metrics


def _predict_probs(trainer: Trainer, dataset: torch.utils.data.Dataset) -> np.ndarray:
    pred = trainer.predict(dataset)
    logits = np.asarray(pred.predictions)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    return probs[:, 1]


def _pick_threshold_max_f1(*, y_true: np.ndarray, p_pos: np.ndarray) -> float:
    thresholds = np.unique(np.round(p_pos, 6))
    thresholds = np.concatenate(([0.0], thresholds, [1.0]))

    best: tuple[float, float] | None = None  # (f1, threshold)
    for t in thresholds:
        y_pred = (p_pos >= t).astype(int)
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        cand = (f1, float(t))
        if best is None:
            best = cand
            continue
        if cand[0] > best[0] + 1e-12:
            best = cand
        elif abs(cand[0] - best[0]) <= 1e-12 and cand[1] > best[1] + 1e-12:
            best = cand

    return best[1] if best is not None else 0.5


def _pick_threshold_for_precision(
    *,
    y_true: np.ndarray,
    p_pos: np.ndarray,
    precision_target: float,
) -> tuple[float, dict[str, float]]:
    thresholds = np.unique(np.round(p_pos, 6))
    thresholds = np.concatenate(([0.0], thresholds, [1.0]))

    best: tuple[float, float, float, float] | None = None
    # (f1, precision, recall, threshold)

    for t in thresholds:
        y_pred = (p_pos >= t).astype(int)
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        if prec + 1e-12 < precision_target:
            continue
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        cand: tuple[float, float, float, float] = (f1, prec, rec, float(t))
        if best is None:
            best = cand
            continue
        if cand[0] > best[0] + 1e-12:
            best = cand
        elif abs(cand[0] - best[0]) <= 1e-12:
            if cand[1] > best[1] + 1e-12:
                best = cand
            elif abs(cand[1] - best[1]) <= 1e-12 and cand[3] > best[3] + 1e-12:
                best = cand

    if best is None:
        return 0.5, {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    f1, prec, rec, t = best
    return t, {"precision": float(prec), "recall": float(rec), "f1": float(f1)}


def _write_predictions_csv(
    *,
    out_csv: Path,
    examples: list[Example],
    p_tech: np.ndarray,
    threshold: float,
    run_id: str,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "docdb_family_id",
                "split",
                "subtype_true",
                "p_technology",
                "subtype_pred",
                "threshold",
                "run_id",
            ],
        )
        w.writeheader()
        for ex, p in sorted(zip(examples, p_tech), key=lambda t: t[0].docdb_family_id):
            w.writerow(
                {
                    "docdb_family_id": ex.docdb_family_id,
                    "split": ex.split,
                    "subtype_true": ID2LABEL[ex.y],
                    "p_technology": float(p),
                    "subtype_pred": ID2LABEL[int(p >= threshold)],
                    "threshold": threshold,
                    "run_id": run_id,
                }
            )


def _confusion_md(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return (
        "| | pred=variety | pred=technology |\n"
        "|---|---:|---:|\n"
        f"| true=variety | {tn} | {fp} |\n"
        f"| true=technology | {fn} | {tp} |\n"
    )


@dataclass
class RunConfig:
    run_id: str
    created_at_utc: str
    git_commit: str | None

    labels_csv: str
    splits_csv: str

    base_model: str
    max_length: int

    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    fp16: bool
    bf16: bool
    learning_rate: float
    weight_decay: float
    num_train_epochs: float
    warmup_ratio: float

    precision_target: float | None
    seed_str: str
    seed_int: int

    device: str
    python: str
    platform: str
    torch: str
    transformers: str


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Train a reproducible variety-vs-technology classifier on the 600 labeled DOCDB families. "
            "Uses only label=yes rows with subtype in {variety, technology}."
        )
    )
    ap.add_argument(
        "--labels-csv",
        type=Path,
        default=Path("metadata/labeling_candidates_600.csv"),
        help="Labeled CSV (committed).",
    )
    ap.add_argument(
        "--splits-csv",
        type=Path,
        default=Path("metadata/labeling_splits_420_90_90.csv"),
        help="Split manifest CSV (committed).",
    )
    ap.add_argument(
        "--base-model",
        type=str,
        default="mpi-inno-comp/paecter",
        help="Hugging Face model id.",
    )
    ap.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help=(
            "Pass trust_remote_code=True to Transformers when loading the model/tokenizer. "
            "Only use this for model repos you trust."
        ),
    )
    ap.add_argument(
        "--use-fast-tokenizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer fast tokenizer when available (default: true).",
    )
    ap.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Tokenizer max length (truncation/padding).",
    )
    ap.add_argument(
        "--train-batch-size",
        type=int,
        default=8,
        help="Per-device train batch size.",
    )
    ap.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate gradients over this many steps.",
    )
    ap.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing (lower memory, slower).",
    )
    ap.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
        help="Per-device eval batch size.",
    )
    ap.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Enable fp16 mixed precision.",
    )
    ap.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Enable bf16 mixed precision (typically not available on MPS).",
    )
    ap.add_argument(
        "--mps-empty-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On Apple MPS, run gc + torch.mps.empty_cache() after eval/save.",
    )
    ap.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="AdamW learning rate.",
    )
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay.",
    )
    ap.add_argument(
        "--epochs",
        type=float,
        default=8.0,
        help="Number of epochs.",
    )
    ap.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.06,
        help="Warmup ratio.",
    )
    ap.add_argument(
        "--precision-target",
        type=float,
        default=None,
        help=(
            "Optional: pick decision threshold on val to reach at least this precision for 'technology'. "
            "If omitted, chooses the threshold that maximizes F1 on val."
        ),
    )
    ap.add_argument(
        "--seed",
        type=str,
        default="2026-01-30",
        help="Seed string used for full reproducibility.",
    )
    ap.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id; default is timestamp-based.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/variety_technology_2026-01-30"),
        help="Model output directory (gitignored).",
    )
    ap.add_argument(
        "--report-md",
        type=Path,
        default=Path("metadata/variety_technology_report_2026-01-30.md"),
        help="Output report markdown (committed).",
    )
    ap.add_argument(
        "--config-json",
        type=Path,
        default=Path("metadata/variety_technology_config_2026-01-30.json"),
        help="Write run config JSON (committed).",
    )
    ap.add_argument(
        "--pred-val-csv",
        type=Path,
        default=Path("metadata/variety_technology_predictions_val_2026-01-30.csv"),
        help="Write validation predictions CSV (committed).",
    )
    ap.add_argument(
        "--pred-test-csv",
        type=Path,
        default=Path("metadata/variety_technology_predictions_test_2026-01-30.csv"),
        help="Write test predictions CSV (committed).",
    )
    ap.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("metadata/variety_technology_metrics_2026-01-30.json"),
        help=(
            "Write machine-readable metrics JSON containing threshold + val/test metrics (committed)."
        ),
    )

    args = ap.parse_args()

    default_output_dir = Path("models/variety_technology_2026-01-30")
    default_report_md = Path("metadata/variety_technology_report_2026-01-30.md")
    default_config_json = Path("metadata/variety_technology_config_2026-01-30.json")
    default_pred_val_csv = Path("metadata/variety_technology_predictions_val_2026-01-30.csv")
    default_pred_test_csv = Path("metadata/variety_technology_predictions_test_2026-01-30.csv")
    default_metrics_json = Path("metadata/variety_technology_metrics_2026-01-30.json")

    if args.base_model != "mpi-inno-comp/paecter":
        model_slug = _slugify_model_id(args.base_model)
        if args.output_dir == default_output_dir:
            args.output_dir = Path(f"models/variety_technology_{model_slug}_{args.seed}")
        if args.report_md == default_report_md:
            args.report_md = Path(f"metadata/variety_technology_{model_slug}_report_{args.seed}.md")
        if args.config_json == default_config_json:
            args.config_json = Path(f"metadata/variety_technology_{model_slug}_config_{args.seed}.json")
        if args.pred_val_csv == default_pred_val_csv:
            args.pred_val_csv = Path(
                f"metadata/variety_technology_{model_slug}_predictions_val_{args.seed}.csv"
            )
        if args.pred_test_csv == default_pred_test_csv:
            args.pred_test_csv = Path(
                f"metadata/variety_technology_{model_slug}_predictions_test_{args.seed}.csv"
            )
        if args.metrics_json == default_metrics_json:
            args.metrics_json = Path(
                f"metadata/variety_technology_{model_slug}_metrics_{args.seed}.json"
            )

    if not args.labels_csv.exists():
        raise SystemExit(f"Not found: {args.labels_csv}")
    if not args.splits_csv.exists():
        raise SystemExit(f"Not found: {args.splits_csv}")
    if args.gradient_accumulation_steps <= 0:
        raise SystemExit("--gradient-accumulation-steps must be > 0")
    if args.fp16 and args.bf16:
        raise SystemExit("Choose at most one of --fp16 or --bf16")
    if _is_mps() and args.bf16:
        raise SystemExit("--bf16 is not supported on MPS; use --fp16 or omit mixed precision")
    if args.precision_target is not None and not (0.0 < args.precision_target <= 1.0):
        raise SystemExit("--precision-target must be in (0, 1]")

    run_id = args.run_id or f"variety_technology_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    seed_int = _stable_seed_int(args.seed)
    _set_all_seeds(seed_int)

    split_by_family = _read_split_manifest(args.splits_csv)
    examples = _read_variety_technology_examples(args.labels_csv, split_by_family)
    if not examples:
        raise SystemExit(
            "No training examples found. Expected label=yes rows with subtype in {variety, technology}."
        )

    counts = _split_counts(examples)

    train_ex = [e for e in examples if e.split == "train"]
    val_ex = [e for e in examples if e.split == "val"]
    test_ex = [e for e in examples if e.split == "test"]

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=bool(args.use_fast_tokenizer),
        trust_remote_code=bool(args.trust_remote_code),
    )

    # Length audit must mirror training truncation behavior.
    lengths = [
        len(
            tokenizer.encode(
                e.text,
                add_special_tokens=True,
                truncation=True,
                max_length=args.max_length,
            )
        )
        for e in examples
    ]
    trunc_rate = float(np.mean([l > args.max_length for l in lengths]))
    length_p50 = int(np.percentile(lengths, 50))
    length_p90 = int(np.percentile(lengths, 90))
    length_p99 = int(np.percentile(lengths, 99))

    def _tokenize(batch: list[Example]) -> TokenizedDataset:
        texts = [e.text for e in batch]
        enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )
        labels = [e.y for e in batch]
        return TokenizedDataset(encodings=enc, labels=labels)

    train_ds = _tokenize(train_ex)
    val_ds = _tokenize(val_ex)
    test_ds = _tokenize(test_ex)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        trust_remote_code=bool(args.trust_remote_code),
    )

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    output_dir = args.output_dir
    checkpoints_dir = output_dir / "checkpoints"
    hf_export_dir = output_dir / "hf_export"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    hf_export_dir.mkdir(parents=True, exist_ok=True)

    metric_for_best_model = "pr_auc"
    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        seed=seed_int,
        data_seed=seed_int,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        fp16=bool(args.fp16),
        bf16=bool(args.bf16),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=25,
        dataloader_pin_memory=not _is_mps(),
        report_to=[],
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
        processing_class=tokenizer,
        callbacks=[
            _DeviceCacheCleanupCallback(enabled=bool(_is_mps() and args.mps_empty_cache))
        ],
    )

    train_result = trainer.train()
    _ = train_result

    trainer.save_model(str(hf_export_dir))
    tokenizer.save_pretrained(str(hf_export_dir))

    val_p_tech = _predict_probs(trainer, val_ds)
    y_val = np.array([e.y for e in val_ex], dtype=int)

    threshold_method = "max_f1"
    if args.precision_target is not None:
        threshold_method = f"precision>={args.precision_target}"
        threshold, val_at_t = _pick_threshold_for_precision(
            y_true=y_val, p_pos=val_p_tech, precision_target=float(args.precision_target)
        )
    else:
        threshold = _pick_threshold_max_f1(y_true=y_val, p_pos=val_p_tech)
        y_val_pred = (val_p_tech >= threshold).astype(int)
        val_at_t = {
            "precision": float(precision_score(y_val, y_val_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_val_pred, zero_division=0)),
            "f1": float(f1_score(y_val, y_val_pred, zero_division=0)),
        }

    test_p_tech = _predict_probs(trainer, test_ds)
    y_test = np.array([e.y for e in test_ex], dtype=int)

    def _metrics_at_threshold(y_true: np.ndarray, p_pos: np.ndarray, t: float) -> dict[str, float]:
        y_pred = (p_pos >= t).astype(int)
        out = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        if len(set(y_true.tolist())) == 2:
            out["roc_auc"] = float(roc_auc_score(y_true, p_pos))
            out["pr_auc"] = float(average_precision_score(y_true, p_pos))
        return out

    val_metrics = _metrics_at_threshold(y_val, val_p_tech, threshold)
    test_metrics = _metrics_at_threshold(y_test, test_p_tech, threshold)

    _write_predictions_csv(
        out_csv=args.pred_val_csv,
        examples=val_ex,
        p_tech=val_p_tech,
        threshold=threshold,
        run_id=run_id,
    )
    _write_predictions_csv(
        out_csv=args.pred_test_csv,
        examples=test_ex,
        p_tech=test_p_tech,
        threshold=threshold,
        run_id=run_id,
    )

    cfg = RunConfig(
        run_id=run_id,
        created_at_utc=_utc_now(),
        git_commit=_git_commit(),
        labels_csv=str(args.labels_csv),
        splits_csv=str(args.splits_csv),
        base_model=args.base_model,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        fp16=bool(args.fp16),
        bf16=bool(args.bf16),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        precision_target=args.precision_target,
        seed_str=args.seed,
        seed_int=seed_int,
        device=_device_summary(),
        python=platform.python_version(),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        torch=torch.__version__,
        transformers=__import__("transformers").__version__,
    )

    args.config_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)

    with args.config_json.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)
        f.write("\n")

    metrics_payload = {
        "run_id": run_id,
        "created_at_utc": cfg.created_at_utc,
        "git_commit": cfg.git_commit,
        "base_model": cfg.base_model,
        "max_length": cfg.max_length,
        "threshold_method": threshold_method,
        "precision_target": cfg.precision_target,
        "threshold_technology": float(threshold),
        "split_counts": counts,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "pred_val_csv": str(args.pred_val_csv),
        "pred_test_csv": str(args.pred_test_csv),
        "report_md": str(args.report_md),
        "hf_export_dir": str(hf_export_dir),
    }
    with args.metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    y_val_pred = (val_p_tech >= threshold).astype(int)
    y_test_pred = (test_p_tech >= threshold).astype(int)

    report = []
    report.append(f"# Variety vs technology report ({cfg.run_id})\n")
    report.append(f"- Created (UTC): {cfg.created_at_utc}")
    report.append(f"- Git commit: {cfg.git_commit or 'unknown'}")
    report.append(f"- Base model: {cfg.base_model}")
    report.append(f"- Device: {cfg.device}")
    report.append(f"- Max length: {cfg.max_length} (truncation + padding=max_length)")
    report.append(f"- Threshold selection: {threshold_method}")
    if cfg.precision_target is not None:
        report.append(f"- Precision target (technology): {cfg.precision_target}")
    report.append(f"- Chosen threshold (technology): {threshold:.6f}")
    report.append(f"- Seed: {cfg.seed_str} (seed_int={cfg.seed_int})\n")

    report.append("## Data\n")
    report.append(
        "Split counts (variety/technology): "
        f"train {counts['train']['variety']}/{counts['train']['technology']}, "
        f"val {counts['val']['variety']}/{counts['val']['technology']}, "
        f"test {counts['test']['variety']}/{counts['test']['technology']}\n"
    )

    report.append("## Token length audit\n")
    report.append(f"- Truncation rate (> max_length): {trunc_rate:.3f}")
    report.append(f"- Token lengths: p50={length_p50}, p90={length_p90}, p99={length_p99}\n")

    report.append("## Training args\n")
    report.append("```json")
    report.append(json.dumps(asdict(cfg), indent=2, sort_keys=True))
    report.append("```\n")

    report.append("## Validation (at chosen threshold)\n")
    report.append("```json")
    report.append(json.dumps(val_metrics, indent=2, sort_keys=True))
    report.append("```\n")
    report.append("Confusion matrix (val):\n")
    report.append(_confusion_md(y_val, y_val_pred) + "\n")

    report.append("## Test (at chosen threshold)\n")
    report.append("```json")
    report.append(json.dumps(test_metrics, indent=2, sort_keys=True))
    report.append("```\n")
    report.append("Confusion matrix (test):\n")
    report.append(_confusion_md(y_test, y_test_pred) + "\n")

    report.append("## Outputs\n")
    report.append(f"- Model (HF-exportable): {hf_export_dir}")
    report.append(f"- Checkpoints: {checkpoints_dir}")
    report.append(f"- Val predictions: {args.pred_val_csv}")
    report.append(f"- Test predictions: {args.pred_test_csv}")

    with args.report_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(report))
        f.write("\n")

    print("=== Done ===")
    print(f"Report: {args.report_md}")
    print(f"Config: {args.config_json}")
    print(f"Metrics: {args.metrics_json}")
    print(f"Model export: {hf_export_dir}")
    print(f"Val metrics: {val_metrics}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
