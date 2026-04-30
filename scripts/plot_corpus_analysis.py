#!/usr/bin/env python3
"""
Generate all figures and LaTeX tables for Section 4: Corpus Analysis.

Reads intermediate CSVs from metadata/corpus_analysis_*.csv and the
applicant harmonization mapping. For the harmonized applicant table,
re-queries the scored parquet via DuckDB.

Outputs:
    paper/figures/growth_trends_counts.pdf   (+ .png)
  paper/figures/growth_trends_indexed.pdf  (+ .png)
  paper/figures/geography_by_subtype.pdf   (+ .png)
  paper/figures/top_applicants_subtype.pdf (+ .png)
  paper/figures/internationalization.pdf   (+ .png)
  paper/tables/corpus_overview.tex
  paper/tables/geographic_specialization.tex
  paper/tables/top_applicants_harmonized.tex
  paper/tables/classifier_uplift_cpc.tex

Usage:
    .venv/bin/python scripts/plot_corpus_analysis.py
"""

import pathlib
import re
import textwrap
import unicodedata

import duckdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
META = ROOT / "metadata"
FIGS = ROOT / "paper" / "figures"
TABS = ROOT / "paper" / "tables"
FIGS.mkdir(parents=True, exist_ok=True)
TABS.mkdir(parents=True, exist_ok=True)

RAW = ROOT / "data" / "raw" / "global_corpus_publevel_plus.parquet"
SCORED = ROOT / "data" / "derived" / (
    "family_modeling_corpus_en_complete"
    "__paecter_precision_seed2026-01-29__FULL__with_subtype.parquet"
)
HARM = META / "applicant_harmonization.csv"
HARM_REVIEW = META / "corpus_analysis_applicant_harmonization_review.csv"

# ── Matplotlib style (APA-like figure typography) ───────────────────────────
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

APA_LABEL_FS = 11
APA_TICK_FS = 9
APA_LEGEND_FS = 9
APA_ANNOTATION_FS = 9
APA_TITLE_FS = 11

# Match figure text to manuscript main text size.
MAIN_TEXT_FIG_FS = 10
APA_LABEL_FS = MAIN_TEXT_FIG_FS
APA_LEGEND_FS = MAIN_TEXT_FIG_FS
APA_TICK_FS = 9
APA_ANNOTATION_FS = 9
APA_TITLE_FS = MAIN_TEXT_FIG_FS


LEGAL_SUFFIX_TOKENS = {
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY",
    "LTD", "LIMITED", "LLC", "PLC", "LP", "LLP", "BV", "NV", "AG",
    "SA", "GMBH", "KG", "KK", "BVBA", "SAS", "SPA", "SRL", "OY",
    "AB", "AS", "PTY",
}


def normalize_applicant_name(name: str | None) -> str:
    """Normalize applicant names deterministically before curated matching."""
    if name is None or pd.isna(name):
        return ""
    normalized = unicodedata.normalize("NFKD", str(name))
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.upper().strip()
    normalized = normalized.replace("&", " AND ")
    normalized = normalized.replace("+", " AND ")
    normalized = re.sub(r"[^A-Z0-9]+", " ", normalized)
    tokens = [token for token in normalized.split() if token not in LEGAL_SUFFIX_TOKENS]
    return " ".join(tokens)


def compute_hhi(counts: pd.Series) -> float:
    shares = counts / counts.sum()
    return round(float((shares ** 2).sum() * 10000), 2)

# ── Load CSVs ────────────────────────────────────────────────────────────────
h1 = pd.read_csv(META / "corpus_analysis_h1_growth_trends.csv")
h3 = pd.read_csv(META / "corpus_analysis_h3_geography.csv")
h4 = pd.read_csv(META / "corpus_analysis_h4_internationalization.csv")
h7 = pd.read_csv(META / "corpus_analysis_h7_geography_by_subtype.csv")
harm = pd.read_csv(HARM)
harm["applicant_normalized"] = harm["applicant_upper"].map(normalize_applicant_name)

conflicting_harm = harm.groupby("applicant_normalized").agg(
    harmonized_n=("harmonized_name", "nunique"),
    entity_n=("entity_type", "nunique"),
).reset_index()
conflicting_harm = conflicting_harm[
    (conflicting_harm["applicant_normalized"] != "")
    & ((conflicting_harm["harmonized_n"] > 1) | (conflicting_harm["entity_n"] > 1))
]
if not conflicting_harm.empty:
    raise ValueError(
        "Conflicting normalized applicant mappings in applicant_harmonization.csv: "
        + ", ".join(conflicting_harm["applicant_normalized"].head(10))
    )

# Global plant share for location quotients
TOTAL_FAMILIES = h1["all_families"].sum()
TOTAL_PLANT = h1["plant_families"].sum()
GLOBAL_PLANT_SHARE = TOTAL_PLANT / TOTAL_FAMILIES

# ══════════════════════════════════════════════════════════════════════════════
# Figure: Growth trends – indexed to 2000 = 100
# ══════════════════════════════════════════════════════════════════════════════
print("Figure: Growth trends (indexed) ...")

fig, ax = plt.subplots(figsize=(7.5, 5.0))

base_year = 2000
cols = {
    "all_families": ("All patent families", "#888888", "-", 2.0),
    "plant_families": ("Plant-related", "#2ca02c", "--", 2.2),
    "technology_families": ("Technology", "#d62728", "-.", 2.0),
    "variety_families": ("Variety", "#1f77b4", ":", 2.2),
}

for col, (label, color, ls, lw) in cols.items():
    base = h1.loc[h1.year == base_year, col].values[0]
    idx = 100.0 * h1[col] / base
    ax.plot(h1.year, idx, label=label, color=color, linestyle=ls, linewidth=lw)

ax.axhline(100, color="grey", linewidth=0.5, zorder=0)
ax.set_xlabel("Priority year", fontsize=10)
ax.set_ylabel("Index (2000 = 100)", fontsize=10)
ax.set_xlim(1985, 2023)
ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
ax.tick_params(axis="both", labelsize=8)
ax.legend(loc="upper left", frameon=False, fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
fig.tight_layout()

for fmt in ("pdf", "png"):
    fig.savefig(FIGS / f"growth_trends_indexed.{fmt}")
plt.close(fig)
print("  -> growth_trends_indexed.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Figure: Growth trends – annual family counts
# ══════════════════════════════════════════════════════════════════════════════
print("Figure: Growth trends (counts) ...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 2.95), sharex=True)
fig.subplots_adjust(wspace=0.22)

ax1.plot(h1.year, h1.all_families, label="All patent families", color="#888888", linewidth=2.2)
ax1.plot(h1.year, h1.plant_families, label="Plant-related", color="#2ca02c", linestyle="--", linewidth=2.2)
ax1.set_xlabel("Priority year", fontsize=9)
ax1.set_ylabel("Families per year", fontsize=9)
ax1.set_xlim(1985, 2023)
ax1.xaxis.set_major_locator(mticker.MultipleLocator(10))
ax1.tick_params(axis="both", labelsize=8)
ax1.legend(loc="upper left", frameon=False, fontsize=7.5)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

ax2.plot(h1.year, h1.variety_families, label="Variety", color="#1f77b4", linestyle=":", linewidth=2.2)
ax2.plot(h1.year, h1.technology_families, label="Technology", color="#d62728", linestyle="-.", linewidth=2.2)
ax2.set_xlabel("Priority year", fontsize=9)
ax2.set_ylabel("Families per year", fontsize=9)
ax2.set_xlim(1985, 2023)
ax2.xaxis.set_major_locator(mticker.MultipleLocator(10))
ax2.tick_params(axis="both", labelsize=8)
ax2.legend(loc="upper left", frameon=False, fontsize=7.5)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()

for fmt in ("pdf", "png"):
    fig.savefig(FIGS / f"growth_trends_counts.{fmt}")
plt.close(fig)
print("  -> growth_trends_counts.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Figure: Geographic composition by subtype (H7)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure: Geography by subtype ...")

# Top-10 jurisdictions by plant total, exclude WO (not a jurisdiction)
h7_plot = h7[h7.auth != "WO"].head(10).copy()
h7_plot = h7_plot.iloc[::-1].reset_index(drop=True)  # reverse for horizontal bars

fig, ax = plt.subplots(figsize=(5.2, 5.1))
y = np.arange(len(h7_plot))
bar_h = 0.34

ax.barh(y, h7_plot.variety_share_pct, height=bar_h, color="#1f77b4", label="Variety")
ax.barh(
    y,
    h7_plot.tech_share_pct,
    left=h7_plot.variety_share_pct,
    height=bar_h,
    color="#d62728",
    label="Technology",
)

# Add n labels on the right
for i, row in h7_plot.iterrows():
    ax.text(
        103,
        i,
        f"$n={row.plant_total:,.0f}$",
        va="center",
        fontsize=13,
    )

ax.set_yticks(y)
ax.set_yticklabels(h7_plot.auth, fontsize=13)
ax.set_xlabel(r"Share of plant-related families (\%)", fontsize=14)
ax.set_xlim(0, 118)
ax.set_xticks(np.arange(0, 101, 20))
ax.tick_params(axis="x", labelsize=13)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.19),
    ncol=2,
    borderaxespad=0.0,
    frameon=False,
    fontsize=13,
)
fig.subplots_adjust(bottom=0.30)

for fmt in ("pdf", "png"):
    fig.savefig(FIGS / f"geography_by_subtype.{fmt}")
plt.close(fig)
print("  -> geography_by_subtype.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Figure: Top applicants by subtype (dual panel, harmonized)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure: Top applicants by subtype ...")

# Re-query parquet with harmonization
con = duckdb.connect()

con.sql(f"""
CREATE OR REPLACE TABLE analysis_base AS
WITH pub AS (
    SELECT *,
        YEAR(earliest_priority_or_filing_date::DATE) AS filing_year
    FROM read_parquet('{RAW}')
),
scored AS (
    SELECT docdb_family_id, label_pred, subtype_pred
    FROM read_parquet('{SCORED}')
),
fam AS (
    SELECT
        p.docdb_family_id,
        MIN(p.filing_year) AS filing_year,
        MAX(s.label_pred) AS plant_pred,
        MAX(s.subtype_pred) AS subtype_pred,
        COUNT(DISTINCT p.publn_auth) AS family_scope,
        MAX(p.has_ep) AS has_ep,
        MAX(p.has_wo) AS has_wo,
        FIRST(p.applicant_names ORDER BY p.appln_id)
            FILTER (WHERE p.applicant_names IS NOT NULL AND p.applicant_names != '')
            AS applicant_names_raw,
        FIRST(p.cpc_list ORDER BY p.appln_id)
            FILTER (WHERE p.cpc_list IS NOT NULL AND p.cpc_list != '')
            AS cpc_list,
        FIRST(p.ipc_list ORDER BY p.appln_id)
            FILTER (WHERE p.ipc_list IS NOT NULL AND p.ipc_list != '')
            AS ipc_list,
        MAX(CASE WHEN p.publn_auth = 'US' AND TRIM(p.publn_kind) IN ('P','P1','P2','P3','P4','P9') THEN 1 ELSE 0 END) AS has_us_plant_patent,
        MAX(CASE WHEN TRIM(p.appln_kind) = 'U' THEN 1 ELSE 0 END) AS has_utility_model
    FROM pub p
    LEFT JOIN scored s ON p.docdb_family_id = s.docdb_family_id
    GROUP BY p.docdb_family_id
)
SELECT
    f.*,
    CASE
        WHEN f.applicant_names_raw LIKE '%;%'
        THEN TRIM(SPLIT_PART(f.applicant_names_raw, ';', 1))
        ELSE TRIM(f.applicant_names_raw)
    END AS first_applicant,
    CASE WHEN f.plant_pred = 'yes' THEN 1 ELSE 0 END AS is_plant,
    CASE WHEN f.subtype_pred = 'variety' THEN 1 ELSE 0 END AS is_variety,
    CASE WHEN f.subtype_pred = 'technology' THEN 1 ELSE 0 END AS is_technology,
    CASE WHEN f.cpc_list LIKE '%A01H%' OR f.ipc_list LIKE '%A01H%' THEN 1 ELSE 0 END AS has_a01h
FROM fam f
WHERE f.filing_year BETWEEN 1985 AND 2023
""")

# Build harmonized applicant counts with deterministic normalization
applicant_base = con.sql("""
SELECT
    docdb_family_id,
    first_applicant,
    is_plant,
    is_variety,
    is_technology
FROM analysis_base
WHERE first_applicant IS NOT NULL AND first_applicant != ''
""").df()
applicant_base["first_applicant_clean"] = applicant_base["first_applicant"].astype(str).str.strip()
applicant_base["first_applicant_upper"] = applicant_base["first_applicant_clean"].str.upper()
applicant_base["applicant_normalized"] = applicant_base["first_applicant_clean"].map(normalize_applicant_name)

harm_lookup = harm[["applicant_normalized", "harmonized_name", "entity_type"]].drop_duplicates()
applicant_base = applicant_base.merge(harm_lookup, on="applicant_normalized", how="left")
applicant_base["applicant"] = applicant_base["harmonized_name"].fillna(applicant_base["first_applicant_upper"])
applicant_base["entity_type"] = applicant_base["entity_type"].fillna("unknown")
applicant_base["harmonization_status"] = np.where(
    applicant_base["harmonized_name"].notna(), "matched", "unmatched"
)

app_harmonized = (
    applicant_base.loc[applicant_base["is_plant"] == 1]
    .groupby(["applicant", "entity_type"], as_index=False)
    .agg(
        plant_count=("is_plant", "sum"),
        variety_count=("is_variety", "sum"),
        technology_count=("is_technology", "sum"),
    )
    .sort_values("plant_count", ascending=False)
)
app_harmonized.to_csv(META / "corpus_analysis_applicants_harmonized.csv", index=False)

harm_review = (
    applicant_base.loc[
        (applicant_base["is_plant"] == 1)
        & (applicant_base["harmonization_status"] == "unmatched")
    ]
    .groupby(["first_applicant_upper", "applicant_normalized"], as_index=False)
    .agg(
        plant_count=("is_plant", "sum"),
        variety_count=("is_variety", "sum"),
        technology_count=("is_technology", "sum"),
    )
    .sort_values(["plant_count", "technology_count", "variety_count"], ascending=False)
)
harm_review.to_csv(HARM_REVIEW, index=False)

# Top-10 variety applicants
top_var = app_harmonized.sort_values("variety_count", ascending=False).head(10).copy()
top_var = top_var.iloc[::-1].reset_index(drop=True)

# Top-10 technology applicants
top_tech = app_harmonized.sort_values("technology_count", ascending=False).head(10).copy()
top_tech = top_tech.iloc[::-1].reset_index(drop=True)

entity_colors = {"corporate": "#1f77b4", "academic": "#2ca02c", "government": "#ff7f0e", "unknown": "#888888"}

TOP_APP_LABEL_FS = 20
TOP_APP_TICK_FS = 18
TOP_APP_LEGEND_FS = 17
TOP_APP_TITLE_FS = 20

def tex_escape(s: str) -> str:
    """Escape special LaTeX characters for usetex labels."""
    for ch, repl in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")]:
        s = s.replace(ch, repl)
    return s

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 8.4), constrained_layout=True)
fig.set_constrained_layout_pads(wspace=0.12, hspace=0.02)

# Variety panel
bar_colors = [entity_colors.get(t, "#888888") for t in top_var.entity_type]
ax1.barh(range(len(top_var)), top_var.variety_count, color=bar_colors, height=0.78)
ax1.set_yticks(range(len(top_var)))
ax1.set_yticklabels([tex_escape(a) for a in top_var.applicant], fontsize=TOP_APP_TICK_FS)
ax1.set_xlabel("Variety families", fontsize=TOP_APP_LABEL_FS)
ax1.tick_params(axis="x", labelsize=TOP_APP_TICK_FS)
ax1.tick_params(axis="y", length=0, pad=2)
ax1.set_title("Variety", fontsize=TOP_APP_TITLE_FS, pad=8)

# Technology panel
bar_colors = [entity_colors.get(t, "#888888") for t in top_tech.entity_type]
ax2.barh(range(len(top_tech)), top_tech.technology_count, color=bar_colors, height=0.78)
ax2.set_yticks(range(len(top_tech)))
ax2.set_yticklabels([tex_escape(a) for a in top_tech.applicant], fontsize=TOP_APP_TICK_FS)
ax2.set_xlabel("Technology families", fontsize=TOP_APP_LABEL_FS)
ax2.tick_params(axis="x", labelsize=TOP_APP_TICK_FS)
ax2.tick_params(axis="y", length=0, pad=2)
ax2.set_title("Technology", fontsize=TOP_APP_TITLE_FS, pad=8)

# Legend
from matplotlib.patches import Patch
legend_items = [
    Patch(facecolor="#1f77b4", label="Corporate"),
    Patch(facecolor="#2ca02c", label="Academic"),
    Patch(facecolor="#ff7f0e", label="Government"),
]
ax1.legend(
    handles=legend_items,
    loc="lower right",
    bbox_to_anchor=(0.98, 0.10),
    ncol=1,
    frameon=False,
    fontsize=TOP_APP_LEGEND_FS,
    handlelength=1.4,
    borderaxespad=0.0,
)

for fmt in ("pdf", "png"):
    fig.savefig(FIGS / f"top_applicants_subtype.{fmt}")
plt.close(fig)
print("  -> top_applicants_subtype.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Figure: Internationalization trends (H4)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure: Internationalization ...")

# Cap at 2020 to avoid truncation bias
h4_plot = h4[h4.year <= 2020].copy()

fig, ax = plt.subplots(figsize=(7.5, 5))

ax.plot(
    h4_plot.year, h4_plot.avg_scope_all,
    label="All families", color="#888888", linestyle="-", linewidth=2.0,
)
ax.plot(
    h4_plot.year, h4_plot.avg_scope_plant,
    label="Plant-related", color="#2ca02c", linestyle="--", linewidth=2.2,
)

ax.set_xlabel("Priority/filing year", fontsize=APA_LABEL_FS)
ax.set_ylabel("Avg. publication jurisdictions per family", fontsize=APA_LABEL_FS)
ax.set_xlim(1985, 2020)
ax.tick_params(axis="both", labelsize=APA_TICK_FS)
ax.legend(loc="upper right", frameon=False, fontsize=APA_LEGEND_FS)

for fmt in ("pdf", "png"):
    fig.savefig(FIGS / f"internationalization.{fmt}")
plt.close(fig)
print("  -> internationalization.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# LATEX TABLES
# ══════════════════════════════════════════════════════════════════════════════

def write_tex(path: pathlib.Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).strip() + "\n")
    print(f"  -> {path.name}")


# ── Table: Corpus Overview ───────────────────────────────────────────────────
print("\nLaTeX tables ...")

# Compute summary metrics from h1 sums + h5 + h9
total_fam = int(h1["all_families"].sum())
plant_fam = int(h1["plant_families"].sum())
var_fam = int(h1["variety_families"].sum())
tech_fam = int(h1["technology_families"].sum())

# Read scope and instrument values from the analysis output
# Re-query for precise values from the family table
scope = con.sql("""
SELECT
    COUNT(*) AS n,
    ROUND(AVG(family_scope), 2) AS avg_scope
FROM analysis_base
""").fetchone()
scope_plant = con.sql("""
SELECT
    COUNT(*) AS n,
    ROUND(AVG(family_scope), 2) AS avg_scope
FROM analysis_base WHERE is_plant = 1
""").fetchone()
scope_var = con.sql("""
SELECT
    COUNT(*) AS n,
    ROUND(AVG(family_scope), 2) AS avg_scope
FROM analysis_base WHERE is_variety = 1
""").fetchone()
scope_tech = con.sql("""
SELECT
    COUNT(*) AS n,
    ROUND(AVG(family_scope), 2) AS avg_scope
FROM analysis_base WHERE is_technology = 1
""").fetchone()

pp_all = con.sql("""
SELECT ROUND(100.0 * SUM(has_us_plant_patent) / COUNT(*), 1) AS pct
FROM analysis_base
""").fetchone()[0]
pp_plant = con.sql("""
SELECT ROUND(100.0 * SUM(has_us_plant_patent) / COUNT(*), 1) AS pct
FROM analysis_base WHERE is_plant = 1
""").fetchone()[0]
pp_var = con.sql("""
SELECT ROUND(100.0 * SUM(has_us_plant_patent) / COUNT(*), 1) AS pct
FROM analysis_base WHERE is_variety = 1
""").fetchone()[0]
pp_tech = con.sql("""
SELECT ROUND(100.0 * SUM(has_us_plant_patent) / COUNT(*), 1) AS pct
FROM analysis_base WHERE is_technology = 1
""").fetchone()[0]
um_all = con.sql("""
SELECT ROUND(100.0 * SUM(has_utility_model) / COUNT(*), 1) AS pct
FROM analysis_base
""").fetchone()[0]
um_plant = con.sql("""
SELECT ROUND(100.0 * SUM(has_utility_model) / COUNT(*), 1) AS pct
FROM analysis_base WHERE is_plant = 1
""").fetchone()[0]
um_var = con.sql("""
SELECT ROUND(100.0 * SUM(has_utility_model) / COUNT(*), 1) AS pct
FROM analysis_base WHERE is_variety = 1
""").fetchone()[0]
um_tech = con.sql("""
SELECT ROUND(100.0 * SUM(has_utility_model) / COUNT(*), 1) AS pct
FROM analysis_base WHERE is_technology = 1
""").fetchone()[0]
inv_all = round(100.0 - pp_all - um_all, 1)
inv_plant = round(100.0 - pp_plant - um_plant, 1)
inv_var = round(100.0 - pp_var - um_var, 1)
inv_tech = round(100.0 - pp_tech - um_tech, 1)

var_share = round(100.0 * var_fam / plant_fam, 1)
tech_share = round(100.0 * tech_fam / plant_fam, 1)

write_tex(TABS / "corpus_overview.tex", f"""\
    \\begin{{table*}}[!t]
    \\centering
    \\small
    \\caption{{Family-level corpus composition and patent instrument mix (1985--2023)}}
    \\label{{tab:corpus_overview}}
    \\setlength{{\\tabcolsep}}{{3pt}}
    \\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}} l S[table-format=6.0] S[table-format=2.1] S[table-format=2.1] S[table-format=1.1] S[table-format=1.2] @{{}}}}
    \\toprule
    Category & {{Families}} & {{Invention patents (\\%)}} & {{US Plant Patents (\\%)}} & {{Utility models (\\%)}} & {{Average scope}} \\\\
    \\midrule
    All families   & {total_fam:>7} & {inv_all} & {pp_all} & {um_all} & {scope[1]} \\\\
    Plant-related  & {plant_fam:>7} & {inv_plant} & {pp_plant} & {um_plant} & {scope_plant[1]} \\\\
    \\quad Variety  & {var_fam:>7} & {inv_var} & {pp_var} & {um_var} & {scope_var[1]} \\\\
    \\quad Technology & {tech_fam:>7} & {inv_tech} & {pp_tech} & {um_tech} & {scope_tech[1]} \\\\
    \\bottomrule
    \\end{{tabular*}}

    \\vspace{{4pt}}
    \\parbox{{\\textwidth}}{{\\footnotesize
    \\textit{{Note:}} US Plant Patents are families with at least one United States publication with kind code
    P, P1, P2, P3, P4, or P9, corresponding to patents granted under 35~USC~\\S\\,161. Utility models are identified
    from PATSTAT application kind U. Invention patents are all remaining families. Scope is the number of distinct
    publication jurisdictions per family. Variety and Technology are mutually exclusive subtypes of plant-related
    families.}}
    \\end{{table*}}
""")


# ── Table: Geographic Specialization (H3) ────────────────────────────────────
h3_tab = h3.head(12).copy()
h3_tab["lq"] = h3_tab["plant_share_pct"] / (100 * GLOBAL_PLANT_SHARE)

geo_rows = []
for _, r in h3_tab.iterrows():
    geo_rows.append(
        f"    {r.auth} & {int(r.all_families):,} & {int(r.plant_families):,} "
        f"& {r.plant_share_pct:.1f} & {r.lq:.2f} & {r.auth_share_of_plant_pct:.1f} \\\\" 
    )
geo_body = "\n".join(geo_rows)

write_tex(TABS / "geographic_specialization.tex", f"""\
    \\begin{{table*}}[!tbp]
    \\centering
    \\small
    \\caption{{Geographic specialization in plant-related patenting (top-12 jurisdictions by plant family count)}}
    \\label{{tab:geographic_specialization}}
    \\setlength{{\\tabcolsep}}{{3pt}}
    \\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}} l r r S[table-format=2.1] S[table-format=1.2] S[table-format=2.1] @{{}}}}
    \\toprule
    Auth. & {{All fam.}} & {{Plant fam.}} & {{Plant (\\%)}} & {{LQ}} & {{\\% of plant}} \\\\
    \\midrule
{geo_body}
    \\bottomrule
    \\end{{tabular*}}

    \\vspace{{4pt}}
    \\parbox{{\\textwidth}}{{\\footnotesize
    \\textit{{Note:}} Auth.\\ = primary application authority (earliest filing).
    Plant (\\%) = share of the authority's families that are plant-related.
    LQ = location quotient (authority plant share / global plant share of {100*GLOBAL_PLANT_SHARE:.1f}\\%).
    \\% of plant = authority's share of all plant-related families.
    WO = international PCT filings; EP = European Patent Office filings.}}
    \\end{{table*}}
""")


# ── Table: Top Applicants Harmonized ──────────────────────────────────────────
top_app = app_harmonized.head(15).copy().reset_index(drop=True)

app_rows = []
for i, r in top_app.iterrows():
    sector = r.entity_type.capitalize()
    name = tex_escape(r.applicant)
    app_rows.append(
        f"    {i+1} & {name} & {int(r.plant_count):,} "
        f"& {int(r.variety_count):,} & {int(r.technology_count):,} & {sector} \\\\"
    )
app_body = "\n".join(app_rows)

write_tex(TABS / "top_applicants_harmonized.tex", f"""\
    \\begin{{table*}}[!tbp]
    \\centering
    \\small
    \caption{{Top-15 applicants in plant-related patenting (harmonized corporate groups, 1985--2023)}}
    \\label{{tab:top_applicants}}
    \\begin{{tabular*}}{{0.92\\textwidth}}{{@{{\\extracolsep{{\\fill}}}} r l r r r l @{{}}}}
    \\toprule
    Rank & Applicant & {{Plant}} & {{Variety}} & {{Technology}} & Sector \\\\
    \\midrule
{app_body}
    \\bottomrule
    \\end{{tabular*}}

    \\vspace{{4pt}}
    \\parbox{{0.92\\textwidth}}{{\\footnotesize
    \\textit{{Note:}} Corporate groups harmonized to current parent where major acquisitions occurred
    during the study period (e.g., Bayer acquired Monsanto in 2018; Corteva spun off from
    DowDuPont in 2019 and includes Pioneer Hi-Bred and Agrigenetics legacy entities; Syngenta
    entities consolidated). Matching first applies deterministic normalization of case,
    punctuation, diacritics, and common legal suffixes before exact lookup in the curated
    harmonization crosswalk; see Appendix~\\ref{{app:applicant_harmonization}} for the
    harmonization rules. Academic and government institutions are abbreviated for space.
    HHI (Herfindahl--Hirschman Index) for harmonized plant-related applicants: {compute_hhi(app_harmonized['plant_count']):.0f};
    overall patent corpus HHI: {compute_hhi(applicant_base.groupby('applicant').size()):.0f}.}}
    \\end{{table*}}
""")


# ── Table: Classification-code composition + uplift (H10) ────────────────────
analysis_total, plant_total = con.sql("""
SELECT COUNT(*) AS all_families, SUM(is_plant) AS plant_families
FROM analysis_base
""").fetchone()

h10_sections = con.sql(f"""
WITH classified AS (
    SELECT
        REGEXP_EXTRACT(
            COALESCE(NULLIF(TRIM(cpc_list), ''), NULLIF(TRIM(ipc_list), '')),
            '([A-HY])',
            1
        ) AS primary_section,
        is_plant
    FROM analysis_base
),
counts AS (
    SELECT
        primary_section AS section,
        COUNT(*) AS all_families,
        SUM(is_plant) AS plant_families
    FROM classified
    WHERE primary_section IS NOT NULL AND primary_section != ''
    GROUP BY primary_section
)
SELECT
    section,
    plant_families,
    ROUND(100.0 * plant_families / {plant_total}, 2) AS plant_pct,
    all_families,
    ROUND(100.0 * all_families / {analysis_total}, 2) AS all_pct
FROM counts
WHERE plant_families > 0
ORDER BY plant_families DESC, section
""").df()
h10_sections.to_csv(META / "corpus_analysis_h10_primary_sections.csv", index=False)

h10_class_families = con.sql(f"""
WITH classified AS (
    SELECT
        REGEXP_EXTRACT(
            COALESCE(NULLIF(TRIM(cpc_list), ''), NULLIF(TRIM(ipc_list), '')),
            '([A-HY][0-9]{{2}}[A-Z])',
            1
        ) AS primary_class_family,
        is_plant
    FROM analysis_base
),
counts AS (
    SELECT
        primary_class_family AS class_family,
        COUNT(*) AS all_families,
        SUM(is_plant) AS plant_families
    FROM classified
    WHERE primary_class_family IS NOT NULL AND primary_class_family != ''
    GROUP BY primary_class_family
)
SELECT
    class_family,
    plant_families,
    ROUND(100.0 * plant_families / {plant_total}, 2) AS plant_pct,
    all_families,
    ROUND(100.0 * all_families / {analysis_total}, 2) AS all_pct
FROM counts
WHERE plant_families > 0
ORDER BY plant_families DESC, class_family
LIMIT 10
""").df()
h10_class_families.to_csv(META / "corpus_analysis_h10_primary_class_families.csv", index=False)

uplift_data = con.sql("""
SELECT
    SUM(is_plant) AS plant_total,
    SUM(CASE WHEN has_a01h = 1 THEN 1 ELSE 0 END) AS all_with_a01h,
    SUM(CASE WHEN is_plant = 1 AND has_a01h = 1 THEN 1 ELSE 0 END) AS with_a01h,
    SUM(CASE WHEN is_plant = 0 AND has_a01h = 1 THEN 1 ELSE 0 END) AS nonplant_with_a01h,
    SUM(CASE WHEN is_plant = 1 AND has_a01h = 0 THEN 1 ELSE 0 END) AS without_a01h,
    ROUND(100.0 * SUM(CASE WHEN is_plant = 1 AND has_a01h = 0 THEN 1 ELSE 0 END)
        / SUM(is_plant), 1) AS uplift_pct
FROM analysis_base
""").fetchone()
pd.DataFrame([{
    "plant_total": int(uplift_data[0]),
    "all_with_a01h": int(uplift_data[1]),
    "with_a01h": int(uplift_data[2]),
    "nonplant_with_a01h": int(uplift_data[3]),
    "without_a01h": int(uplift_data[4]),
    "uplift_pct": uplift_data[5],
}]).to_csv(META / "corpus_analysis_h10_a01h_uplift.csv", index=False)

section_rows = []
for _, r in h10_sections.iterrows():
    section_rows.append(
        f"    {r.section} & {int(r.plant_families):,} & {r.plant_pct:.1f} "
        f"& {int(r.all_families):,} & {r.all_pct:.1f} \\\\"
    )
section_body = "\n".join(section_rows)

class_rows = []
for _, r in h10_class_families.iterrows():
    class_rows.append(
        f"    {r.class_family} & {int(r.plant_families):,} & {r.plant_pct:.1f} "
        f"& {int(r.all_families):,} & {r.all_pct:.1f} \\\\"
    )
class_body = "\n".join(class_rows)

write_tex(TABS / "classifier_uplift_cpc.tex", f"""\
    \\begin{{table*}}[!tbp]
    \\centering
    \\small
    \\caption{{IPC/CPC composition of plant-related families and classifier coverage beyond A01H}}
    \\label{{tab:classifier_uplift}}
    \\setlength{{\\tabcolsep}}{{4pt}}
    \\renewcommand{{\\arraystretch}}{{0.94}}
    \\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}} l r S[table-format=2.1] r S[table-format=2.1] @{{}}}}
    \\toprule
    \\multicolumn{{5}}{{l}}{{\\textit{{Panel A: Primary IPC/CPC section composition}}}} \\\\
    \\midrule
    Section & {{\\shortstack{{Plant\\\\families}}}} & {{\\shortstack{{Share of\\\\plant (\\%)}}}} & {{\\shortstack{{All\\\\families}}}} & {{\\shortstack{{Share of\\\\retrieved (\\%)}}}} \\\\
    \\midrule
{section_body}
    \\midrule
    \\multicolumn{{5}}{{l}}{{\\textit{{Panel B: Top primary class families by plant-family count}}}} \\\\
    \\midrule
    Class family & {{\\shortstack{{Plant\\\\families}}}} & {{\\shortstack{{Share of\\\\plant (\\%)}}}} & {{\\shortstack{{All\\\\families}}}} & {{\\shortstack{{Share of\\\\retrieved (\\%)}}}} \\\\
    \\midrule
{class_body}
    \\midrule
    \\multicolumn{{5}}{{l}}{{\\textit{{Panel C: Classifier uplift relative to A01H}}}} \\\\
    \\midrule
    \\multicolumn{{1}}{{l}}{{Metric}} & \\multicolumn{{1}}{{r}}{{Families}} & \\multicolumn{{1}}{{r}}{{Share of A01H-coded (\\%)}} & \\multicolumn{{2}}{{c}}{{Share of plant (\\%)}} \\\\
    \\midrule
    Plant-related families with A01H code & {int(uplift_data[2]):,} & {100*uplift_data[2]/uplift_data[1]:.1f} & \\multicolumn{{2}}{{c}}{{{100*uplift_data[2]/uplift_data[0]:.1f}}} \\\\
    Non-plant families with A01H code & {int(uplift_data[3]):,} & {100*uplift_data[3]/uplift_data[1]:.1f} & \\multicolumn{{2}}{{c}}{{n/a}} \\\\
    Plant-related families without A01H code & {int(uplift_data[4]):,} & \\multicolumn{{1}}{{c}}{{n/a}} & \\multicolumn{{2}}{{c}}{{{uplift_data[5]:.1f}}} \\\\
    \\bottomrule
    \\end{{tabular*}}

    \\parbox{{\\textwidth}}{{\\scriptsize
    \\textit{{Note:}} Panels~A and B assign each DOCDB family a primary classification using the
    first CPC symbol, falling back to the first IPC symbol when CPC is absent.
    Panel~A reports primary section composition; Panel~B reports the top-10 primary
    class families (Letter + two digits + Letter) ranked by plant-family count.
    A01H = ``New plants or non-transgenic processes for obtaining them; plant reproduction by tissue culture techniques.''
    Panel~C counts A01H anywhere in the family's CPC or IPC lists rather than only as the
    primary class family. ``Share of A01H-coded'' uses all A01H-coded families as the
    denominator, and ``Share of plant'' uses all classifier-identified plant families.
    ``n/a'' indicates that the denominator is not applicable for that row. \\% of retrieved
    uses the full retrieved corpus as denominator.}}
    \\renewcommand{{\\arraystretch}}{{1}}
    \\end{{table*}}
""")


# ── Compute and report CAGRs for the LaTeX prose ─────────────────────────────
print("\n=== Key statistics for LaTeX prose ===")
for period, y0, y1 in [("1985--2023", 1985, 2023), ("2000--2023", 2000, 2023)]:
    r0 = h1[h1.year == y0].iloc[0]
    r1 = h1[h1.year == y1].iloc[0]
    n = y1 - y0
    for col in ["all_families", "plant_families", "variety_families", "technology_families"]:
        if r0[col] > 0:
            cagr = (r1[col] / r0[col]) ** (1 / n) - 1
            print(f"  {period} CAGR {col}: {cagr * 100:.2f}%")

# HHI harmonized
hhi_harm = compute_hhi(app_harmonized["plant_count"])
print(f"  Harmonized plant HHI: {hhi_harm}")

# Tech CAGR 2010-2023
r_2010 = h1[h1.year == 2010].iloc[0]
r_2023 = h1[h1.year == 2023].iloc[0]
cagr_tech = (r_2023["technology_families"] / r_2010["technology_families"]) ** (1/13) - 1
cagr_var = (r_2023["variety_families"] / r_2010["variety_families"]) ** (1/13) - 1
print(f"  2010--2023 CAGR technology: {cagr_tech*100:.2f}%")
print(f"  2010--2023 CAGR variety: {cagr_var*100:.2f}%")

# Total applicant counts by sector
sector_counts = (
    applicant_base.loc[applicant_base["is_plant"] == 1]
    .groupby("entity_type", as_index=False)
    .agg(
        variety_fam=("is_variety", "sum"),
        tech_fam=("is_technology", "sum"),
    )
    .rename(columns={"entity_type": "sector"})
    .sort_values("tech_fam", ascending=False)
)
print("\n  Sector breakdown:")
print(sector_counts.to_string())

con.close()

print("\n=== DONE ===")
