from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Merge translation results JSONL files (e.g., Apple Translation + NLLB fallback) into a single join-compatible JSONL. "
            "Merges by source_text_sha256; prefers successful translations."
        )
    )
    p.add_argument(
        "--primary",
        default="metadata/family_translation_results.jsonl",
        help="Primary results JSONL (typically Apple Translation).",
    )
    p.add_argument(
        "--secondary",
        default="metadata/family_translation_results_nllb.jsonl",
        help="Secondary results JSONL (e.g., NLLB fallback).",
    )
    p.add_argument(
        "--output",
        default="metadata/family_translation_results_combined.jsonl",
        help="Output merged results JSONL.",
    )
    p.add_argument(
        "--prefer",
        choices=["primary", "secondary"],
        default="primary",
        help="If both have ok=true for the same sha, which one wins (default: primary).",
    )
    return p.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    args = parse_args()
    primary = Path(args.primary)
    secondary = Path(args.secondary)
    output = Path(args.output)

    if not primary.exists():
        raise SystemExit(f"Not found: {primary}")
    if not secondary.exists():
        raise SystemExit(f"Not found: {secondary}")

    by_key: dict[tuple[int, str], dict] = {}

    def add_row(row: dict, source: str) -> None:
        fam = row.get("docdb_family_id")
        field = row.get("field")
        if fam is None or field is None:
            return
        key = (int(fam), str(field))

        existing = by_key.get(key)
        if existing is None:
            by_key[key] = row
            return

        ex_ok = bool(existing.get("ok"))
        row_ok = bool(row.get("ok"))

        # Prefer successful translations.
        if ex_ok and not row_ok:
            return
        if row_ok and not ex_ok:
            by_key[key] = row
            return

        # Both ok or both not ok: tie-break by preference.
        if source == "primary" and args.prefer == "primary":
            by_key[key] = row
        elif source == "secondary" and args.prefer == "secondary":
            by_key[key] = row

    for row in iter_jsonl(primary):
        add_row(row, source="primary")

    for row in iter_jsonl(secondary):
        add_row(row, source="secondary")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for _key, row in by_key.items():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Merged rows: {len(by_key):,}")
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
