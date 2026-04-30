#!/usr/bin/env python
"""
Build a new PINTO gold test from the 2026-03-03 PINTO export.

Steps:
1. Map all 106 PINTO patents to DOCDB families (99 by publn_nr, 3 more by appln_nr)
2. Pick representative publication per family (longest text, then largest appln_id)
3. Optionally add documented negatives from the 600-family labeled set
4. Save reproducibility parquet + summary CSV

Output: metadata/gold_test_pinto_2026-03-03_reproducibility.parquet
        metadata/gold_test_pinto_2026-03-03_summary.csv
"""

import pandas as pd
import duckdb

# ── 1. Load & normalise PINTO export ──────────────────────────────
df = pd.read_excel('data/raw/pinto_export_03-03-2026.xlsx')
actual_cols = ['species_variety', 'variety_denomination', 'patent_title',
               'patent_number', 'patent_holder', 'patent_link']
df.columns = actual_cols

patents = df[['patent_number', 'patent_link']].drop_duplicates(subset='patent_number')
patents['publn_auth'] = patents['patent_number'].str.extract(r'^([A-Z]{2})')
patents['publn_nr'] = patents['patent_number'].str.extract(r'^[A-Z]{2}(\d+)')
# Extract application number from EPO Register URLs
patents['appln_nr_from_url'] = patents['patent_link'].str.extract(r'number=[A-Z]{2}(\d+)')

print(f'Unique PINTO patents: {len(patents)}')

# ── 2. Match to enriched corpus ──────────────────────────────────
con = duckdb.connect()

# Register enriched corpus
enriched_path = 'data/derived/global_corpus_publevel_plus_text_enriched_apple_nllb.parquet'
con.execute(f"CREATE VIEW enriched AS SELECT * FROM read_parquet('{enriched_path}')")

# Match by publication number
con.execute("CREATE TABLE pinto_patents AS SELECT * FROM patents")
publn_matches = con.execute("""
    SELECT DISTINCT p.patent_number, e.docdb_family_id
    FROM pinto_patents p
    JOIN enriched e
        ON CAST(e.publn_nr AS VARCHAR) = p.publn_nr
        AND e.publn_auth = p.publn_auth
""").fetchdf()

matched_by_publn = set(publn_matches['patent_number'])
print(f'Matched by publication number: {len(matched_by_publn)} patents → {publn_matches["docdb_family_id"].nunique()} families')

# Match remaining by application number
unmatched = patents[~patents['patent_number'].isin(matched_by_publn)].copy()
unmatched = unmatched[unmatched['appln_nr_from_url'].notna()]
if len(unmatched) > 0:
    con.execute("CREATE TABLE unmatched_patents AS SELECT * FROM unmatched")
    appln_matches = con.execute("""
        SELECT DISTINCT u.patent_number, e.docdb_family_id
        FROM unmatched_patents u
        JOIN enriched e
            ON CAST(e.appln_nr AS VARCHAR) = u.appln_nr_from_url
    """).fetchdf()
    print(f'Matched by application number: {len(appln_matches)} patents → {appln_matches["docdb_family_id"].nunique()} families')
else:
    appln_matches = pd.DataFrame(columns=['patent_number', 'docdb_family_id'])

# Combine all matches
all_matches = pd.concat([publn_matches, appln_matches], ignore_index=True).drop_duplicates(subset='docdb_family_id')
pinto_family_ids = sorted(all_matches['docdb_family_id'].unique())
print(f'\nTotal PINTO DOCDB families: {len(pinto_family_ids)}')

# ── 3. Select representative publication per family ──────────────
# Rule: longest title+abstract text, then largest appln_id (matches export_family_modeling_corpus_en.py)
family_id_list = ','.join(str(f) for f in pinto_family_ids)

gold_pos = con.execute(f"""
    WITH ranked AS (
        SELECT *,
            LENGTH(COALESCE(appln_title_en_final, '')) + LENGTH(COALESCE(appln_abstract_en_final, '')) AS text_len,
            ROW_NUMBER() OVER (
                PARTITION BY docdb_family_id
                ORDER BY
                    LENGTH(COALESCE(appln_title_en_final, '')) + LENGTH(COALESCE(appln_abstract_en_final, '')) DESC,
                    appln_id DESC
            ) AS rn
        FROM enriched
        WHERE docdb_family_id IN ({family_id_list})
    )
    SELECT * FROM ranked WHERE rn = 1
""").fetchdf()

gold_pos['gold_label'] = 1
gold_pos['gold_source'] = 'pinto_export_2026-03-03'
gold_pos['gold_confidence'] = 'high'
gold_pos['gold_notes'] = 'PINTO database export (variety-patent pairs with EPO Register links)'
print(f'Representative publications for positives: {len(gold_pos)} families')

# ── 4. Add documented negatives from 600-family labeled set ──────
labeled = pd.read_csv('metadata/labeling_candidates_600.csv')
neg_labeled = labeled[labeled['label'].str.lower() == 'no'].copy()
print(f'\n600-family labeled set negatives: {len(neg_labeled)}')

# Verify no overlap with PINTO positives
neg_family_ids = set(neg_labeled['docdb_family_id'].astype(str))
pos_family_ids = set(str(f) for f in pinto_family_ids)
overlap = neg_family_ids.intersection(pos_family_ids)
print(f'Overlap between negatives and PINTO positives: {len(overlap)}')

# Sample negatives to roughly match positive count (balanced gold set)
n_neg = len(gold_pos)  # match positive count
if len(neg_labeled) >= n_neg:
    # Use a deterministic hash-based sample
    neg_family_id_list = ','.join(str(int(f)) for f in neg_labeled['docdb_family_id'])
    gold_neg = con.execute(f"""
        WITH neg_families AS (
            SELECT docdb_family_id FROM (VALUES {','.join(f'({int(f)})' for f in neg_labeled['docdb_family_id'])}) AS t(docdb_family_id)
        ),
        ranked AS (
            SELECT e.*,
                LENGTH(COALESCE(e.appln_title_en_final, '')) + LENGTH(COALESCE(e.appln_abstract_en_final, '')) AS text_len,
                ROW_NUMBER() OVER (
                    PARTITION BY e.docdb_family_id
                    ORDER BY
                        LENGTH(COALESCE(e.appln_title_en_final, '')) + LENGTH(COALESCE(e.appln_abstract_en_final, '')) DESC,
                        e.appln_id DESC
                ) AS rn
            FROM enriched e
            JOIN neg_families n ON e.docdb_family_id = n.docdb_family_id
        ),
        reps AS (
            SELECT * FROM ranked WHERE rn = 1
        )
        SELECT * FROM reps
        ORDER BY md5(CAST(docdb_family_id AS VARCHAR) || '2026-03-03')
        LIMIT {n_neg}
    """).fetchdf()
    gold_neg['gold_label'] = 0
    gold_neg['gold_source'] = 'manual_labels_600_set'
    gold_neg['gold_confidence'] = 'high'
    gold_neg['gold_notes'] = 'From 600-family labeled set (label=no, documented labeling pipeline)'
    print(f'Sampled {len(gold_neg)} negatives (balanced with {len(gold_pos)} positives)')
else:
    gold_neg = pd.DataFrame()
    print(f'WARNING: Only {len(neg_labeled)} negatives available; using all')

# ── 5. Combine and save ──────────────────────────────────────────
# Save positive-only version
cols_keep = [c for c in gold_pos.columns if c not in ('rn', 'text_len')]
gold_pos_out = gold_pos[cols_keep].copy()
gold_pos_out.to_parquet('metadata/gold_test_pinto_2026-03-03_positives_only.parquet', compression='zstd')
print(f'\nSaved positive-only parquet: {len(gold_pos_out)} families')

# Save balanced version (positives + negatives)
if len(gold_neg) > 0:
    gold_neg_out = gold_neg[[c for c in gold_neg.columns if c in cols_keep]].copy()
    gold_balanced = pd.concat([gold_pos_out, gold_neg_out], ignore_index=True)
    gold_balanced.to_parquet('metadata/gold_test_pinto_2026-03-03_reproducibility.parquet', compression='zstd')
    print(f'Saved balanced parquet: {len(gold_balanced)} families ({(gold_balanced["gold_label"]==1).sum()} pos, {(gold_balanced["gold_label"]==0).sum()} neg)')
    
    # Summary CSV
    summary = gold_balanced[['docdb_family_id', 'publn_auth', 'publn_nr', 'gold_label',
                              'appln_title_en_final', 'appln_abstract_en_final',
                              'gold_source', 'gold_confidence', 'gold_notes']].copy()
    summary.columns = ['family_id', 'publn_auth', 'publn_nr', 'gold_label',
                        'title_en_final', 'abstract_en_final',
                        'gold_source', 'gold_confidence', 'gold_notes']
    summary.to_csv('metadata/gold_test_pinto_2026-03-03_summary.csv', index=False)
    print(f'Saved summary CSV: metadata/gold_test_pinto_2026-03-03_summary.csv')

# Also log the PINTO patent → family mapping
mapping_log = all_matches.merge(patents[['patent_number', 'publn_auth', 'publn_nr', 'appln_nr_from_url']], on='patent_number')
mapping_log.to_csv('metadata/pinto_2026-03-03_patent_family_mapping.csv', index=False)
print(f'Saved patent→family mapping: metadata/pinto_2026-03-03_patent_family_mapping.csv')

# ── 6. Print comparison with old gold test ────────────────────────
print('\n' + '='*60)
print('COMPARISON WITH OLD GOLD TEST')
print('='*60)
old_gold = pd.read_csv('metadata/gold_test_complete_summary_2026-03-02.csv')
old_pos = set(old_gold[old_gold['gold_label']==1]['family_id'].astype(str))
new_pos = set(str(f) for f in pinto_family_ids)

print(f'Old PINTO positives: {len(old_pos)}')
print(f'New PINTO positives: {len(new_pos)}')
print(f'Overlap: {len(old_pos & new_pos)}')
print(f'Only in old: {len(old_pos - new_pos)}')
print(f'Only in new: {len(new_pos - old_pos)}')
print(f'\nOld negatives: {(old_gold["gold_label"]==0).sum()} (source: Lens.org, undocumented)')
if len(gold_neg) > 0:
    print(f'New negatives: {len(gold_neg)} (source: 600-family labeled set, documented)')

# Report unmatched PINTO patents
all_matched_patents = set(all_matches['patent_number'])
all_pinto_patents = set(patents['patent_number'])
still_unmatched = all_pinto_patents - all_matched_patents
if still_unmatched:
    print(f'\nUnmatched PINTO patents ({len(still_unmatched)}):')
    for p in sorted(still_unmatched):
        print(f'  {p}')
