#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
import time
from pathlib import Path

# HF Hub defaults to short (10s) timeouts; bump them for robustness.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
# Avoid tqdm progress bars (they spam logs and hide useful progress lines).
os.environ.setdefault("TRANSFORMERS_NO_TQDM", "1")
# Helpful for MPS edge cases.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils.logging import disable_progress_bar


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _threshold_from_report(report_md: Path) -> float:
    text = report_md.read_text(encoding="utf-8")
    m = re.search(
        r"^\s*-\s*Chosen threshold \(technology\):\s*([0-9]*\.?[0-9]+)\s*$",
        text,
        re.MULTILINE,
    )
    if not m:
        raise SystemExit(
            f"Could not find a 'Chosen threshold (technology):' line in {report_md}. "
            "Pass --threshold explicitly instead."
        )
    return float(m.group(1))


@dataclass(frozen=True)
class ColumnSpec:
    docdb_family_id: str
    rep_appln_id: str | None
    text_expr: str


def _infer_column_spec(
    con: duckdb.DuckDBPyConnection, parquet_path: Path, *, text_col: str
) -> ColumnSpec:
    parquet_posix = parquet_path.as_posix().replace("'", "''")
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{parquet_posix}')"
    ).fetchall()
    cols = {r[0] for r in rows}

    if "docdb_family_id" not in cols:
        raise SystemExit(f"Input parquet is missing docdb_family_id: {parquet_path}")

    rep_appln_id = "rep_appln_id" if "rep_appln_id" in cols else None

    if text_col in cols:
        text_expr = text_col
    elif "title_en" in cols and "abstract_en" in cols:
        text_expr = "trim(coalesce(title_en, '') || '\\n\\n' || coalesce(abstract_en, ''))"
    else:
        raise SystemExit(
            f"Input parquet must contain '{text_col}' or ('title_en' and 'abstract_en'). "
            f"Found columns: {sorted(list(cols))[:50]}"
        )

    return ColumnSpec(
        docdb_family_id="docdb_family_id",
        rep_appln_id=rep_appln_id,
        text_expr=text_expr,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Score plant-positive families with a fine-tuned variety-vs-technology classifier and write probabilities. "
            "Reads (a) a family-level parquet with text and (b) a plant-positive parquet with docdb_family_id + label_pred/p_yes."
        )
    )
    ap.add_argument(
        "--family-parquet",
        type=Path,
        default=Path("data/derived/family_modeling_corpus_en_complete.parquet"),
        help="Input family-level parquet with text (local, gitignored).",
    )
    ap.add_argument(
        "--plant-positives-parquet",
        type=Path,
        default=Path(
            "data/derived/family_modeling_corpus_en_complete__bert_for_patents_precision_seed2026-01-29__FULL.parquet"
        ),
        help=(
            "Stage-1 scoring output parquet containing at least docdb_family_id and (label_pred or p_yes). "
            "Used to define the set of families to subtype-score."
        ),
    )
    ap.add_argument(
        "--positives-filter",
        type=str,
        default="label_pred_yes",
        choices=["label_pred_yes", "p_yes_gte"],
        help="How to select plant-positive families from --plant-positives-parquet.",
    )
    ap.add_argument(
        "--positives-threshold",
        type=float,
        default=None,
        help="Only used when --positives-filter p_yes_gte; selects families with p_yes >= this.",
    )
    ap.add_argument(
        "--exclude-families-csv",
        type=Path,
        default=Path("metadata/exclude_families_labeled_600.csv"),
        help="CSV of docdb_family_id values to exclude (default: the labeled 600; prevents leakage).",
    )
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/variety_technology_2026-01-30/hf_export"),
        help="Path to HF-exportable subtype model directory (local, gitignored).",
    )
    ap.add_argument(
        "--out-parquet",
        type=Path,
        default=Path(
            "data/derived/family_modeling_corpus_en_complete__plant_yes__variety_technology_scores.parquet"
        ),
        help="Output parquet (local, gitignored).",
    )
    ap.add_argument(
        "--text-col",
        type=str,
        default="text_en",
        help="Preferred text column to score (fallback: title_en + abstract_en).",
    )
    ap.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Tokenizer max length (truncation).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for model inference.",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional decision threshold to emit subtype_pred=technology/variety (default: omit subtype_pred).",
    )
    ap.add_argument(
        "--threshold-from-report",
        type=Path,
        default=None,
        help="Optional path to a subtype training report markdown; extracts the 'Chosen threshold (technology):' value.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick smoke tests.",
    )
    ap.add_argument(
        "--progress-every-rows",
        type=int,
        default=50000,
        help="Print a progress line every N scored rows (default: 50000).",
    )

    args = ap.parse_args()

    if not args.family_parquet.exists():
        raise SystemExit(f"Not found: {args.family_parquet}")
    if not args.plant_positives_parquet.exists():
        raise SystemExit(f"Not found: {args.plant_positives_parquet}")
    if args.exclude_families_csv is not None and not args.exclude_families_csv.exists():
        raise SystemExit(f"Not found: {args.exclude_families_csv}")
    if not args.model_dir.exists():
        raise SystemExit(f"Not found: {args.model_dir} (did you train/export the model?)")
    if args.max_length <= 0:
        raise SystemExit("--max-length must be > 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be > 0")
    if args.progress_every_rows is not None and args.progress_every_rows <= 0:
        raise SystemExit("--progress-every-rows must be > 0")

    if args.positives_filter == "p_yes_gte" and args.positives_threshold is None:
        raise SystemExit("--positives-threshold is required when --positives-filter p_yes_gte")

    threshold: float | None = args.threshold
    if threshold is None and args.threshold_from_report is not None:
        if not args.threshold_from_report.exists():
            raise SystemExit(f"Not found: {args.threshold_from_report}")
        threshold = _threshold_from_report(args.threshold_from_report)

    device = _device()

    print(f"Scoring at: {_utc_now()}")
    print(f"Device: {device.type}")
    print(f"Family input: {args.family_parquet}")
    print(f"Plant positives: {args.plant_positives_parquet}")
    print(f"Positives filter: {args.positives_filter}")
    if args.positives_threshold is not None:
        print(f"Positives threshold: {args.positives_threshold}")
    print(f"Exclude list: {args.exclude_families_csv}")
    print(f"Model: {args.model_dir}")
    print(f"Output: {args.out_parquet}")
    if threshold is not None:
        print(f"Subtype decision threshold (technology): {threshold}")
    if args.limit is not None:
        print(f"Limit: {args.limit}")

    disable_progress_bar()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir.as_posix(), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir.as_posix())
    model.eval()
    model.to(device)

    con = duckdb.connect(database=":memory:")
    spec = _infer_column_spec(con, args.family_parquet, text_col=args.text_col)

    family_posix = args.family_parquet.as_posix().replace("'", "''")
    positives_posix = args.plant_positives_parquet.as_posix().replace("'", "''")

    with_bits = []
    where_bits = []

    # Plant-positive selection.
    if args.positives_filter == "label_pred_yes":
        where_bits.append("label_pred = 'yes'")
    else:
        where_bits.append(f"p_yes >= {float(args.positives_threshold)}")

    with_bits.append(
        "plant_yes AS ("
        "SELECT CAST(docdb_family_id AS BIGINT) AS docdb_family_id "
        f"FROM read_parquet('{positives_posix}') "
        f"WHERE {' AND '.join(where_bits)}"
        ")"
    )

    exclude_where = ""
    if args.exclude_families_csv is not None:
        exclude_posix = args.exclude_families_csv.as_posix().replace("'", "''")
        with_bits.append(
            "excluded_families AS ("
            "SELECT CAST(docdb_family_id AS BIGINT) AS docdb_family_id "
            f"FROM read_csv_auto('{exclude_posix}')"
            ")"
        )
        exclude_where = (
            "AND docdb_family_id NOT IN (SELECT docdb_family_id FROM excluded_families)"
        )

    with_clause = "WITH " + ", ".join(with_bits)

    rep_select = f", {spec.rep_appln_id}" if spec.rep_appln_id is not None else ""
    limit_clause = f"LIMIT {int(args.limit)}" if args.limit is not None else ""

    q = f"""
    {with_clause}
    SELECT
      docdb_family_id
      {rep_select},
      {spec.text_expr} AS text
    FROM read_parquet('{family_posix}')
    WHERE
      {spec.text_expr} IS NOT NULL
      AND length(trim({spec.text_expr})) > 0
      AND docdb_family_id IN (SELECT docdb_family_id FROM plant_yes)
      {exclude_where}
    ORDER BY docdb_family_id
    {limit_clause}
    """

    reader = con.execute(q).fetch_record_batch(args.batch_size)

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)

    scored_at = _utc_now()

    fields: list[pa.Field] = [
        pa.field("docdb_family_id", pa.int64()),
    ]
    if spec.rep_appln_id is not None:
        fields.append(pa.field("rep_appln_id", pa.int64()))
    fields.append(pa.field("p_technology", pa.float32()))
    if threshold is not None:
        fields.append(pa.field("subtype_pred", pa.string()))
        fields.append(pa.field("threshold", pa.float32()))
    fields.append(pa.field("scored_at_utc", pa.string()))
    schema = pa.schema(fields)

    writer: pq.ParquetWriter | None = None
    n_rows = 0
    n_tech = 0
    started = time.time()
    # Print an initial progress line after the first batch, then every N rows.
    next_progress_at = 0

    try:
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break

            if batch.num_rows == 0:
                continue

            docdb_family_id = batch.column(
                batch.schema.get_field_index("docdb_family_id")
            ).to_numpy(zero_copy_only=False)

            rep_appln_id = None
            if spec.rep_appln_id is not None:
                rep_appln_id = batch.column(
                    batch.schema.get_field_index("rep_appln_id")
                ).to_numpy(zero_copy_only=False)

            texts = batch.column(batch.schema.get_field_index("text")).to_pylist()

            enc = tokenizer(
                texts,
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                out = model(**enc)
                logits = out.logits
                p_tech = (
                    torch.softmax(logits, dim=-1)[:, 1]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

            out_dict: dict[str, pa.Array] = {
                "docdb_family_id": pa.array(docdb_family_id, type=pa.int64()),
                "p_technology": pa.array(p_tech, type=pa.float32()),
            }
            if rep_appln_id is not None:
                out_dict["rep_appln_id"] = pa.array(rep_appln_id, type=pa.int64())

            if threshold is not None:
                preds = np.where(p_tech >= float(threshold), "technology", "variety")
                out_dict["subtype_pred"] = pa.array(preds.tolist(), type=pa.string())
                out_dict["threshold"] = pa.array(
                    np.full(
                        shape=(len(p_tech),),
                        fill_value=float(threshold),
                        dtype=np.float32,
                    ),
                    type=pa.float32(),
                )
                n_tech += int((p_tech >= float(threshold)).sum())

            out_dict["scored_at_utc"] = pa.array([scored_at] * len(p_tech), type=pa.string())

            out_table = pa.Table.from_pydict(out_dict, schema=schema)

            if writer is None:
                writer = pq.ParquetWriter(
                    args.out_parquet.as_posix(),
                    schema=schema,
                    compression="zstd",
                )

            writer.write_table(out_table)
            n_rows += batch.num_rows

            if args.progress_every_rows is not None and n_rows >= next_progress_at:
                elapsed = max(time.time() - started, 1e-9)
                rate = n_rows / elapsed
                if threshold is not None:
                    frac_tech = n_tech / max(n_rows, 1)
                    print(
                        f"Progress: {n_rows:,} rows | technology={n_tech:,} ({frac_tech:.3%}) | {rate:,.1f} rows/s",
                        flush=True,
                    )
                else:
                    print(
                        f"Progress: {n_rows:,} rows | {rate:,.1f} rows/s",
                        flush=True,
                    )
                next_progress_at += int(args.progress_every_rows)

    finally:
        if writer is not None:
            writer.close()

    print("=== Done ===")
    print(f"Rows scored: {n_rows:,}")
    if threshold is not None:
        print(
            f"Predicted technology (p_technology >= {threshold}): {n_tech:,} ({(n_tech / max(n_rows, 1)):.3%})"
        )


if __name__ == "__main__":
    main()
