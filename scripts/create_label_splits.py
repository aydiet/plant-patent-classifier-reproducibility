#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Row:
    docdb_family_id: int
    label: str


def _md5_hex(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _read_rows(labels_csv: Path) -> list[Row]:
    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"docdb_family_id", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing columns in {labels_csv}: {sorted(missing)}")

        rows: list[Row] = []
        for i, r in enumerate(reader, start=2):
            raw_id = (r.get("docdb_family_id") or "").strip()
            raw_label = (r.get("label") or "").strip().lower()
            if not raw_id:
                raise SystemExit(f"Empty docdb_family_id at CSV line {i}")
            if raw_label not in {"yes", "no"}:
                raise SystemExit(
                    f"Unexpected label={raw_label!r} at CSV line {i} (expected 'yes'/'no')"
                )
            rows.append(Row(docdb_family_id=int(raw_id), label=raw_label))

    # Ensure uniqueness (families are the unit of labeling).
    ids = [r.docdb_family_id for r in rows]
    if len(ids) != len(set(ids)):
        # Deterministic debug output: show first few dups.
        seen: set[int] = set()
        dups: list[int] = []
        for id_ in ids:
            if id_ in seen:
                dups.append(id_)
            else:
                seen.add(id_)
            if len(dups) >= 10:
                break
        raise SystemExit(
            f"Duplicate docdb_family_id values in {labels_csv} (e.g. {dups[:10]}). "
            "Fix duplicates before splitting."
        )

    return rows


def _allocate_yes_counts(*, n_yes: int, split_sizes: dict[str, int]) -> dict[str, int]:
    total = sum(split_sizes.values())
    if total <= 0:
        raise SystemExit("Total split size must be > 0")

    # Exact split sizes are enforced; we compute yes-counts per split using
    # floor + largest remainder so counts sum to n_yes deterministically.
    floors: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for split, n in split_sizes.items():
        exact = n_yes * (n / total)
        floor = int(exact)
        floors[split] = floor
        remainders.append((exact - floor, split))

    remaining = n_yes - sum(floors.values())
    if remaining < 0:
        raise SystemExit("Internal error: allocated too many yes labels")

    # Deterministic tie-break: sort by remainder desc, then split name asc.
    remainders.sort(key=lambda t: (-t[0], t[1]))
    out = dict(floors)
    for _, split in remainders:
        if remaining <= 0:
            break
        out[split] += 1
        remaining -= 1

    if sum(out.values()) != n_yes:
        raise SystemExit("Internal error: yes allocation did not sum")

    return out


def _sorted_by_hash(rows: Iterable[Row], *, seed: str) -> list[tuple[str, Row]]:
    keyed: list[tuple[str, Row]] = []
    for r in rows:
        h = _md5_hex(f"{r.docdb_family_id}|{seed}")
        keyed.append((h, r))
    keyed.sort(key=lambda t: (t[0], t[1].docdb_family_id))
    return keyed


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Create deterministic train/val/test splits for the manually-labeled 600-family queue. "
            "Splits are family-level (docdb_family_id) and stratified by label only (yes/no)."
        )
    )
    ap.add_argument(
        "--labels-csv",
        type=Path,
        default=Path("metadata/labeling_candidates_600.csv"),
        help="Input labeled CSV (committable).",
    )
    ap.add_argument(
        "--out-splits-csv",
        type=Path,
        default=Path("metadata/labeling_splits_420_90_90.csv"),
        help="Output split manifest CSV (committable).",
    )
    ap.add_argument(
        "--out-exclude-csv",
        type=Path,
        default=Path("metadata/exclude_families_labeled_600.csv"),
        help="Output exclusion list CSV (committable).",
    )
    ap.add_argument("--train", type=int, default=420, help="Train set size.")
    ap.add_argument("--val", type=int, default=90, help="Validation set size.")
    ap.add_argument("--test", type=int, default=90, help="Test set size.")
    ap.add_argument(
        "--seed",
        type=str,
        default="2026-01-28",
        help="Deterministic seed string used only for stable ordering.",
    )

    args = ap.parse_args()

    if not args.labels_csv.exists():
        raise SystemExit(f"Not found: {args.labels_csv}")

    split_sizes = {"train": args.train, "val": args.val, "test": args.test}
    total_target = sum(split_sizes.values())
    if any(n <= 0 for n in split_sizes.values()):
        raise SystemExit("--train/--val/--test must all be > 0")

    rows = _read_rows(args.labels_csv)
    if len(rows) != total_target:
        raise SystemExit(
            f"Expected {total_target} labeled rows for sizes {split_sizes}, got {len(rows)} rows in {args.labels_csv}"
        )

    yes_rows = [r for r in rows if r.label == "yes"]
    no_rows = [r for r in rows if r.label == "no"]

    yes_counts = _allocate_yes_counts(n_yes=len(yes_rows), split_sizes=split_sizes)
    # Enforce exact split sizes.
    no_counts = {split: split_sizes[split] - yes_counts[split] for split in split_sizes}

    yes_sorted = _sorted_by_hash(yes_rows, seed=args.seed)
    no_sorted = _sorted_by_hash(no_rows, seed=args.seed)

    assignments: dict[int, tuple[str, str]] = {}

    def assign_group(group_sorted: list[tuple[str, Row]], counts: dict[str, int]) -> None:
        idx = 0
        for split in ("train", "val", "test"):
            take = counts[split]
            for h, r in group_sorted[idx : idx + take]:
                if r.docdb_family_id in assignments:
                    raise SystemExit(f"Internal error: double assignment for {r.docdb_family_id}")
                assignments[r.docdb_family_id] = (split, h)
            idx += take

    assign_group(yes_sorted, yes_counts)
    assign_group(no_sorted, no_counts)

    if len(assignments) != len(rows):
        raise SystemExit("Internal error: not all rows assigned")

    # Write split manifest.
    args.out_splits_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_splits_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "docdb_family_id",
                "split",
                "label",
                "seed",
                "md5_family_seed",
            ],
        )
        w.writeheader()
        for r in sorted(rows, key=lambda x: x.docdb_family_id):
            split, h = assignments[r.docdb_family_id]
            w.writerow(
                {
                    "docdb_family_id": r.docdb_family_id,
                    "split": split,
                    "label": r.label,
                    "seed": args.seed,
                    "md5_family_seed": h,
                }
            )

    # Write exclusion list.
    args.out_exclude_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_exclude_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["docdb_family_id"])
        w.writeheader()
        for fam_id in sorted(assignments.keys()):
            w.writerow({"docdb_family_id": fam_id})

    # Print a short summary for reproducibility.
    summary: dict[tuple[str, str], int] = {}
    id_to_label = {r.docdb_family_id: r.label for r in rows}
    for fam_id, (split, _) in assignments.items():
        key = (split, id_to_label[fam_id])
        summary[key] = summary.get(key, 0) + 1

    print(f"Read labeled rows: {len(rows):,} (yes={len(yes_rows):,}, no={len(no_rows):,})")
    print(f"Wrote splits: {args.out_splits_csv}")
    print(f"Wrote exclusion list: {args.out_exclude_csv}")
    for split in ("train", "val", "test"):
        y = summary.get((split, "yes"), 0)
        n = summary.get((split, "no"), 0)
        print(f"  {split}: {y+n} (yes={y}, no={n})")


if __name__ == "__main__":
    main()
