# Family translation candidate report

**Input:** `data/raw/global_corpus_publevel_plus.parquet`  
**Output (JSONL):** `metadata/family_translation_candidates.jsonl`  
**Target language:** `en`

This selects one **title** and/or one **abstract** per `docdb_family_id` for translation **only** when the family contains no English for that field.

Selection rule (deterministic): choose the longest available text in the family; tie-breaker by larger `appln_id`.

## Counts

- Families total (distinct): 844,923
- Title candidates (families with no English title): 18,212
- Abstract candidates (families with no English abstract): 20,054
- Total candidate rows: 38,266

## Source language breakdown

These are the source languages present in the candidate set (by field). Apple Translation requires the corresponding language pairs to be installed in System Settings.

### Title candidates

| source_lg | candidates |
|---|---:|
| zh | 6,435 |
| es | 3,299 |
| de | 1,822 |
| pt | 1,739 |
| ru | 1,269 |
| fr | 599 |
| da | 422 |
| no | 386 |
| it | 379 |
| ko | 356 |
| uk | 283 |
| el | 270 |
| ja | 262 |
| tr | 177 |
| sv | 155 |
| fi | 148 |
| nl | 66 |
| id | 32 |
| lt | 22 |
| lv | 22 |
| cs | 17 |
| pl | 15 |
| is | 14 |
| ar | 10 |
| et | 8 |
| ro | 2 |
| sh | 2 |
| sk | 1 |

### Abstract candidates

| source_lg | candidates |
|---|---:|
| zh | 6,452 |
| ko | 4,125 |
| es | 3,409 |
| pt | 1,530 |
| ru | 963 |
| pl | 660 |
| de | 489 |
| fr | 479 |
| uk | 367 |
| el | 299 |
| ja | 274 |
| hu | 239 |
| no | 191 |
| tr | 171 |
| ro | 165 |
| cs | 87 |
| it | 19 |
| fi | 18 |
| sl | 18 |
| sh | 17 |
| bg | 16 |
| nl | 16 |
| ar | 11 |
| hr | 9 |
| lv | 9 |
| sk | 9 |
| sr | 4 |
| sv | 4 |
| da | 2 |
| me | 2 |

## Notes

- Candidate rows are meant to be consumed by a Swift CLI that uses Apple’s Translation framework.
- The CLI should group rows by `source_lg` because Apple batch translation APIs require a consistent source language per batch.
