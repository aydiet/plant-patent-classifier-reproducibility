#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_NUM_RE = re.compile(r"^[-+]?((\d{1,3}(,\d{3})+)|(\d+))(\.\d+)?%?$")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _latex_escape(s: str) -> str:
    # Keep it minimal; most of our content is numeric/codes/model names.
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _looks_numeric(value: str) -> bool:
    value = value.strip()
    if not value:
        return False
    return bool(_NUM_RE.match(value))


def _infer_colspec(headers: list[str], rows: list[list[str]]) -> str:
    n = len(headers)
    if n == 0:
        return "l"

    # First column left; others right if mostly numeric.
    colspec = ["l"]
    for col_idx in range(1, n):
        col_values = [r[col_idx].strip() for r in rows if col_idx < len(r) and r[col_idx].strip()]
        if col_values and sum(_looks_numeric(v) for v in col_values) / len(col_values) >= 0.8:
            colspec.append("r")
        else:
            colspec.append("l")
    return "".join(colspec)


def _split_md_row(line: str) -> list[str]:
    parts = [p.strip() for p in line.strip().split("|")]
    if parts and parts[0] == "":
        parts = parts[1:]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return parts


def _extract_first_md_table(text: str, *, start_after: str | None = None) -> tuple[list[str], list[list[str]]]:
    lines = text.splitlines()
    start_idx = 0
    if start_after:
        for i, line in enumerate(lines):
            if start_after in line:
                start_idx = i + 1
                break

    # Find header line then alignment line.
    for i in range(start_idx, len(lines) - 1):
        header = lines[i]
        align = lines[i + 1]
        if "|" not in header:
            continue
        if not re.search(r"\|\s*:?-{3,}:?\s*\|", align):
            continue

        headers = _split_md_row(header)
        rows: list[list[str]] = []
        for j in range(i + 2, len(lines)):
            row_line = lines[j]
            if "|" not in row_line:
                break
            if row_line.strip().startswith("```"):
                break
            row = _split_md_row(row_line)
            if not row or all(c == "" for c in row):
                break
            rows.append(row)
        return headers, rows

    raise ValueError("No markdown table found")


def _write_latex_table(
    *,
    out_path: Path,
    caption: str,
    label: str,
    headers: list[str],
    rows: list[list[str]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    escaped_headers = [_latex_escape(h) for h in headers]
    escaped_rows = [[_latex_escape(c) for c in r] for r in rows]
    colspec = _infer_colspec(headers, rows)

    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{{_latex_escape(caption)}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")
    lines.append(" ".join([" ".join(escaped_headers[:1]), *["& " + h for h in escaped_headers[1:]]]) + r"\\")
    lines.append(r"\midrule")

    for r in escaped_rows:
        if len(r) < len(escaped_headers):
            r = r + [""] * (len(escaped_headers) - len(r))
        row_line = " ".join([" ".join(r[:1]), *["& " + c for c in r[1:]]]) + r"\\"
        lines.append(row_line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_key_value_table(
    *,
    out_path: Path,
    caption: str,
    label: str,
    items: list[tuple[str, str]],
) -> None:
    headers = ["Metric", "Value"]
    rows = [[k, v] for k, v in items]
    _write_latex_table(out_path=out_path, caption=caption, label=label, headers=headers, rows=rows)


def _parse_int_from_bullet(text: str, key_prefix: str) -> int:
    # Matches lines like: - Families total (distinct): 844,923
    m = re.search(rf"^\s*[-*]\s+{re.escape(key_prefix)}\s*:\s*\*\*(.*?)\*\*\s*$", text, flags=re.M)
    if not m:
        m = re.search(rf"^\s*[-*]\s+{re.escape(key_prefix)}\s*:\s*(.*?)\s*$", text, flags=re.M)
    if not m:
        raise ValueError(f"Could not find bullet for: {key_prefix}")
    raw = m.group(1).strip()
    raw = raw.replace(",", "")
    raw = re.sub(r"[^0-9]", "", raw)
    return int(raw)


def _parse_count_line(text: str, prefix: str) -> int:
    # For code blocks like: records=5317
    m = re.search(rf"^{re.escape(prefix)}\s*=\s*([0-9]+)", text, flags=re.M)
    if not m:
        raise ValueError(f"Missing line: {prefix}=")
    return int(m.group(1))


def _write_numbers_macros(out_path: Path, macros: dict[str, int | float | str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("% Auto-generated. Do not edit by hand.")
    lines.append("% Generated by: scripts/paper_build_assets.py")
    lines.append("")

    for name, val in macros.items():
        if isinstance(val, int):
            lines.append(rf"\newcommand{{\{name}}}{{\num{{{val}}}}}")
        else:
            lines.append(rf"\newcommand{{\{name}}}{{{_latex_escape(str(val))}}}")

    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _extract_query_filters(query_log_md: str) -> dict[str, object]:
    pub_date = None
    m = re.search(r"\* Publication date:\s*([0-9-]+)\s*to\s*([0-9-]+)", query_log_md)
    if m:
        pub_date = (m.group(1), m.group(2))

    def _extract_prefix_list(header: str) -> list[str]:
        # Finds bullet lists under a header like "* CPC prefixes:".
        lines = query_log_md.splitlines()
        start = None
        for i, line in enumerate(lines):
            if line.strip() == header:
                start = i + 1
                break
        if start is None:
            return []

        prefixes: list[str] = []
        for line in lines[start:]:
            if line.startswith("### ") or line.startswith("## "):
                break
            if line.strip() == "":
                continue

            # Stop if we hit another top-level bullet header (e.g., "* IPC prefixes:").
            if line.startswith("*") and line.strip().endswith("prefixes:"):
                break

            # We only want the indented bullet items: "  * A01H*".
            m2 = re.match(r"^\s{2,}\*\s+(.+?)\s*$", line)
            if m2:
                prefixes.append(m2.group(1))
                continue

            # Once we've collected something, stop when the indent block ends.
            if prefixes and not line.startswith("  "):
                break
        return prefixes

    cpc = _extract_prefix_list("* CPC prefixes:")
    ipc = _extract_prefix_list("* IPC prefixes:")

    return {"pub_date": pub_date, "cpc": cpc, "ipc": ipc}


def _write_query_filters_table(*, out_path: Path, pub_date: tuple[str, str] | None, cpc: list[str], ipc: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _as_multiline(items: list[str]) -> str:
        if not items:
            return ""
        # Use a nested tabular to allow line breaks without requiring extra packages.
        escaped = [_latex_escape(x) for x in items]
        body = "\\\\\n".join(escaped)
        return r"\begin{tabular}[t]{@{}l@{}}" + body + r"\end{tabular}"

    pub_date_str = ""
    if pub_date:
        pub_date_str = _latex_escape(f"{pub_date[0]} to {pub_date[1]}")

    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{TIP/PATSTAT query filters used for initial retrieval}")
    lines.append(r"\label{tab:query_filters}")
    lines.append(r"\begin{tabular}{ll}")
    lines.append(r"\toprule")
    lines.append(r"Filter & Value\\")
    lines.append(r"\midrule")
    lines.append(rf"Publication date window & {pub_date_str}\\")
    lines.append(rf"CPC prefixes & {_as_multiline(cpc)}\\")
    lines.append(rf"IPC prefixes & {_as_multiline(ipc)}\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


@dataclass(frozen=True)
class Inputs:
    metadata_dir: Path
    out_tables_dir: Path


def build_assets(inputs: Inputs, *, top_n_codes: int) -> None:
    md = inputs.metadata_dir
    out = inputs.out_tables_dir

    # 0) Query filters
    query_log = _read_text(md / "query_log.md")
    filters = _extract_query_filters(query_log)
    _write_query_filters_table(
        out_path=out / "query_filters.tex",
        pub_date=filters["pub_date"] if isinstance(filters.get("pub_date"), tuple) else None,
        cpc=list(filters.get("cpc", [])),
        ipc=list(filters.get("ipc", [])),
    )

    # 1) Translation candidate report
    translation_report = _read_text(md / "family_translation_candidates_report.md")
    families_total = _parse_int_from_bullet(translation_report, "Families total (distinct)")
    title_candidates = _parse_int_from_bullet(translation_report, "Title candidates (families with no English title)")
    abstract_candidates = _parse_int_from_bullet(translation_report, "Abstract candidates (families with no English abstract)")
    total_candidate_rows = _parse_int_from_bullet(translation_report, "Total candidate rows")

    _write_key_value_table(
        out_path=out / "translation_candidates_counts.tex",
        caption="English-translation candidate set (family-level)",
        label="tab:translation_candidates_counts",
        items=[
            ("Families total (distinct)", f"{families_total:,}"),
            ("Title candidates (no English title)", f"{title_candidates:,}"),
            ("Abstract candidates (no English abstract)", f"{abstract_candidates:,}"),
            ("Total candidate rows", f"{total_candidate_rows:,}"),
        ],
    )

    title_headers, title_rows = _extract_first_md_table(translation_report, start_after="### Title candidates")
    _write_latex_table(
        out_path=out / "translation_title_languages.tex",
        caption="Source language distribution for title translation candidates",
        label="tab:translation_title_languages",
        headers=title_headers,
        rows=title_rows,
    )

    abstract_headers, abstract_rows = _extract_first_md_table(translation_report, start_after="### Abstract candidates")
    _write_latex_table(
        out_path=out / "translation_abstract_languages.tex",
        caption="Source language distribution for abstract translation candidates",
        label="tab:translation_abstract_languages",
        headers=abstract_headers,
        rows=abstract_rows,
    )

    # 2) Plant vs non-plant model comparison
    model_comp = _read_text(md / "model_comparison_2026-01-29.md")
    mc_headers, mc_rows = _extract_first_md_table(model_comp, start_after="## Summary table")
    _write_latex_table(
        out_path=out / "model_comparison_summary.tex",
        caption="Plant vs non-plant: model comparison (validation threshold chosen for precision target)",
        label="tab:model_comparison_summary",
        headers=mc_headers,
        rows=mc_rows,
    )

    # 3) Gold test eval. The "_apples" suffix is a historical run label for
    # the fixed-threshold comparison report used by the manuscript tables.
    gold = _read_text(md / "gold_test_model_comparison_2026-01-30_apples.md")
    gold_headers, gold_rows = _extract_first_md_table(gold, start_after="## Summary table")
    _write_latex_table(
        out_path=out / "gold_test_summary_thresholds.tex",
        caption="Gold test evaluation using each model's training-report threshold",
        label="tab:gold_test_summary_thresholds",
        headers=gold_headers,
        rows=gold_rows,
    )

    gold_fixed_headers, gold_fixed_rows = _extract_first_md_table(gold, start_after="### Fixed threshold = 0.500000")
    _write_latex_table(
        out_path=out / "gold_test_fixed_threshold_050.tex",
        caption="Gold test evaluation at a fixed threshold (0.5) for apples-to-apples comparison",
        label="tab:gold_test_fixed_threshold_050",
        headers=gold_fixed_headers,
        rows=gold_fixed_rows,
    )

    # 4) Variety vs technology. The "_quick" suffix is a historical run label;
    # the generated manuscript table uses the finalized table caption below.
    vt = _read_text(md / "variety_technology_model_comparison_2026-01-30_quick.md")
    vt_ch_headers, vt_ch_rows = _extract_first_md_table(vt, start_after="## At each model's chosen threshold")
    _write_latex_table(
        out_path=out / "variety_tech_at_chosen_threshold.tex",
        caption="Variety vs technology: performance at each model's chosen threshold",
        label="tab:variety_tech_at_chosen_threshold",
        headers=vt_ch_headers,
        rows=vt_ch_rows,
    )

    vt_fixed_headers, vt_fixed_rows = _extract_first_md_table(vt, start_after="## Fixed threshold: 0.5")
    _write_latex_table(
        out_path=out / "variety_tech_fixed_threshold_050.tex",
        caption="Variety vs technology: performance at a fixed threshold (0.5)",
        label="tab:variety_tech_fixed_threshold_050",
        headers=vt_fixed_headers,
        rows=vt_fixed_rows,
    )

    # 5) OPS fulltext
    ops = _read_text(md / "ops_fulltext_results_technology_50000_2026-02-04_v4_summary.md")
    ops_processed = _parse_int_from_bullet(ops, "Processed")
    ops_success = _parse_int_from_bullet(ops, "Success (got any claims or description)")
    ops_both = _parse_int_from_bullet(ops, "Got both claims + description")

    _write_key_value_table(
        out_path=out / "ops_fulltext_headline.tex",
        caption="OPS full-text retrieval headline results (technology subset)",
        label="tab:ops_fulltext_headline",
        items=[
            ("Families processed", f"{ops_processed:,}"),
            ("Families with any claims/description", f"{ops_success:,}"),
            ("Families with both claims and description", f"{ops_both:,}"),
        ],
    )

    # 6) Crop tagging headline + top codes
    crop = _read_text(md / "crop_tagging_rules_section_weighted_report_2026-02-10.md")
    crop_scanned = _parse_int_from_bullet(crop, "Families scanned")
    crop_pred = _parse_int_from_bullet(crop, "Families with >=1 predicted code")
    crop_rows_written = _parse_int_from_bullet(crop, "Prediction rows written")

    _write_key_value_table(
        out_path=out / "crop_tagging_headline.tex",
        caption="Crop tagging headline counts (rules-only, section-weighted)",
        label="tab:crop_tagging_headline",
        items=[
            ("Families scanned", f"{crop_scanned:,}"),
            ("Families with >=1 predicted code", f"{crop_pred:,}"),
            ("Prediction rows written", f"{crop_rows_written:,}"),
        ],
    )

    # Parse top codes list
    top_codes: list[tuple[str, int]] = []
    in_top = False
    for line in crop.splitlines():
        if line.strip() == "## Top codes by families predicted":
            in_top = True
            continue
        if in_top:
            if line.startswith("## ") and "Top codes" not in line:
                break
            m = re.match(r"^\s*-\s*([A-Z0-9_]+)\s*:\s*([0-9,]+)\s*$", line)
            if m:
                code = m.group(1)
                n = int(m.group(2).replace(",", ""))
                top_codes.append((code, n))
    top_codes = top_codes[:top_n_codes]

    _write_latex_table(
        out_path=out / "crop_tagging_top_codes.tex",
        caption=f"Top predicted crop codes by number of families (top {len(top_codes)})",
        label="tab:crop_tagging_top_codes",
        headers=["code", "families"],
        rows=[[c, f"{n:,}"] for c, n in top_codes],
    )

    # 7) Paper numbers macros
    macros: dict[str, int | str] = {
        "TotalFamilies": families_total,
        "TitleTranslationFamilies": title_candidates,
        "AbstractTranslationFamilies": abstract_candidates,
        "TranslationCandidateRows": total_candidate_rows,
        "OpsFamiliesProcessed": ops_processed,
        "OpsFamiliesSuccess": ops_success,
        "OpsFamiliesBoth": ops_both,
        "CropFamiliesScanned": crop_scanned,
        "CropFamiliesTagged": crop_pred,
    }
    _write_numbers_macros(out / "paper_numbers.tex", macros)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX paper tables/macros from committed metadata reports.")
    parser.add_argument("--metadata-dir", type=Path, default=Path("metadata"))
    parser.add_argument("--out-tables-dir", type=Path, default=Path("paper") / "tables")
    parser.add_argument("--top-n-codes", type=int, default=20)
    args = parser.parse_args()

    build_assets(
        Inputs(metadata_dir=args.metadata_dir, out_tables_dir=args.out_tables_dir),
        top_n_codes=args.top_n_codes,
    )


if __name__ == "__main__":
    main()
