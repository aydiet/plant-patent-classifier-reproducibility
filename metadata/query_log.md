## TIP / PATSTAT Query Log

**Project:** Dynamic efficiency of patents in plant breeding
**Author:** Aylish Dietrich
**Platform:** EPO Technology Intelligence Platform (TIP – PATSTAT BigQuery)
**Execution date:** 2026-01-06
**Corpus size:** 3,728,861 publications – 844,923 DOCDB families

### Filters

* Publication date: 1985-01-01 to 2025-12-31
* CPC prefixes:

  * A01H*
  * C07K14/415*
  * C12N5*
  * C12N15*
  * C12N2310/20*
  * C12Q1/68*
  * Y02A40/13*
  * Y02A40/146*
* IPC prefixes:

  * A01H*
  * C07K*
  * C12N5*
  * C12N15*
  * C12Q1/68*

### Text fields

* Title: English preferred, fallback to any language
* Abstract: English preferred, fallback to any language

### Priority definition

* Earliest family priority date via tls204_appln_prior → tls201_appln.appln_filing_date
* Fallback: earliest family filing date

### Legal events

* Aggregated from tls231_inpadoc_legal_event using `event_effective_date`

### SQL

The full TIP/PATSTAT export query and export client are tracked in the replication package:

* SQL:
  * `scripts/tip_patstat_export_global_corpus_publevel_plus.sql`
* TIP export client:
  * `scripts/export_tip_patstat_publevel_plus.py`

This query log records the execution date, parameter values, and expected output counts for auditability.
