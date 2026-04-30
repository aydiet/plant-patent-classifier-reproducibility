WITH
-- A) Publication-level base in the manuscript extraction window (global scope)
pubs AS (
  SELECT
    p.appln_id,
    p.publn_auth, p.publn_nr, p.publn_kind,
    p.publn_date,
    a.appln_auth, a.appln_nr, a.appln_kind,
    a.appln_filing_date,
    a.docdb_family_id
  FROM tls211_pat_publn p
  JOIN tls201_appln a ON a.appln_id = p.appln_id
  WHERE p.publn_date BETWEEN DATE '1985-01-01' AND DATE '2025-12-31'
),

-- B) Families matching CPC/IPC prefixes (subcategories included via LIKE 'prefix%')
hits AS (
  SELECT DISTINCT a.docdb_family_id
  FROM tls201_appln a
  LEFT JOIN tls224_appln_cpc c ON c.appln_id = a.appln_id
  LEFT JOIN tls209_appln_ipc i ON i.appln_id = a.appln_id
  WHERE
    -- CPC (spaces stripped so "C07K 14/415" matches)
    REPLACE(c.cpc_class_symbol,' ','') LIKE 'A01H%' OR
    REPLACE(c.cpc_class_symbol,' ','') LIKE 'C07K14/415%' OR
    REPLACE(c.cpc_class_symbol,' ','') LIKE 'C12N5%' OR
    REPLACE(c.cpc_class_symbol,' ','') LIKE 'C12N15%' OR
    REPLACE(c.cpc_class_symbol,' ','') LIKE 'C12N2310/20%' OR
    REPLACE(c.cpc_class_symbol,' ','') LIKE 'C12Q1/68%' OR
    REPLACE(c.cpc_class_symbol,' ','') LIKE 'Y02A40/13%' OR
    REPLACE(c.cpc_class_symbol,' ','') LIKE 'Y02A40/146%' OR

    -- IPC
    REPLACE(i.ipc_class_symbol,' ','') LIKE 'A01H%' OR
    REPLACE(i.ipc_class_symbol,' ','') LIKE 'C07K%' OR
    REPLACE(i.ipc_class_symbol,' ','') LIKE 'C12N5%' OR
    REPLACE(i.ipc_class_symbol,' ','') LIKE 'C12N15%' OR
    REPLACE(i.ipc_class_symbol,' ','') LIKE 'C12Q1/68%'
),

-- C) Title/abstract fields: English preferred, else fallback to any language + language codes
titles_en AS (
  SELECT appln_id, appln_title AS title_en
  FROM tls202_appln_title
  WHERE appln_title_lg = 'en'
),
titles_any AS (
  SELECT
    appln_id,
    MIN(appln_title) AS title_any,
    MIN(appln_title_lg) AS title_any_lg
  FROM tls202_appln_title
  GROUP BY appln_id
),
abstracts_en AS (
  SELECT appln_id, appln_abstract AS abstract_en
  FROM tls203_appln_abstr
  WHERE appln_abstract_lg = 'en'
),
abstracts_any AS (
  SELECT
    appln_id,
    MIN(appln_abstract) AS abstract_any,
    MIN(appln_abstract_lg) AS abstract_any_lg
  FROM tls203_appln_abstr
  GROUP BY appln_id
),

-- D) CPC/IPC lists per application
cpc_agg AS (
  SELECT
    appln_id,
    STRING_AGG(REPLACE(cpc_class_symbol,' ',''), '; ') AS cpc_list
  FROM tls224_appln_cpc
  GROUP BY appln_id
),
ipc_agg AS (
  SELECT
    appln_id,
    STRING_AGG(REPLACE(ipc_class_symbol,' ',''), '; ') AS ipc_list
  FROM tls209_appln_ipc
  GROUP BY appln_id
),

-- E) Applicants + applicant countries
applicants AS (
  SELECT
    pa.appln_id,
    STRING_AGG(COALESCE(p.person_name,''), '; ') AS applicant_names,
    STRING_AGG(DISTINCT COALESCE(p.person_ctry_code,''), '; ') AS applicant_country_codes
  FROM tls207_pers_appln pa
  JOIN tls206_person p ON p.person_id = pa.person_id
  WHERE pa.applt_seq_nr > 0
  GROUP BY pa.appln_id
),

-- F) Inventors + inventor countries
inventors AS (
  SELECT
    pa.appln_id,
    STRING_AGG(COALESCE(p.person_name,''), '; ') AS inventor_names,
    STRING_AGG(DISTINCT COALESCE(p.person_ctry_code,''), '; ') AS inventor_country_codes
  FROM tls207_pers_appln pa
  JOIN tls206_person p ON p.person_id = pa.person_id
  WHERE pa.invt_seq_nr > 0
  GROUP BY pa.appln_id
),

-- Earliest priority date per family:
-- tls204_appln_prior is a link table; join to the PRIOR application and use its filing date.
earliest_priority AS (
  SELECT
    a.docdb_family_id,
    MIN(pa.appln_filing_date) AS earliest_priority_date
  FROM tls201_appln a
  JOIN tls204_appln_prior pr ON pr.appln_id = a.appln_id
  JOIN tls201_appln pa ON pa.appln_id = pr.prior_appln_id
  GROUP BY a.docdb_family_id
),

-- Fallback: earliest filing date in the family
family_earliest_filing AS (
  SELECT
    docdb_family_id,
    MIN(appln_filing_date) AS family_earliest_filing_date
  FROM tls201_appln
  GROUP BY docdb_family_id
),

-- G) Family authorities list + EP/WO flags (based on publications)
family_auths AS (
  SELECT
    a.docdb_family_id,
    STRING_AGG(DISTINCT p.publn_auth, '; ') AS family_publn_auths,
    MAX(CASE WHEN p.publn_auth = 'EP' THEN 1 ELSE 0 END) AS has_ep,
    MAX(CASE WHEN p.publn_auth = 'WO' THEN 1 ELSE 0 END) AS has_wo
  FROM tls201_appln a
  JOIN tls211_pat_publn p ON p.appln_id = a.appln_id
  GROUP BY a.docdb_family_id
),

-- H) Legal events summary per family (TIP/PATSTAT schema uses event_effective_date)
legal_events AS (
  SELECT
    a.docdb_family_id,
    COUNT(*) AS legal_event_count,
    MIN(le.event_effective_date) AS first_legal_event_date,
    MAX(le.event_effective_date) AS last_legal_event_date,
    STRING_AGG(DISTINCT le.event_code, '; ') AS legal_event_codes
  FROM tls201_appln a
  JOIN tls231_inpadoc_legal_event le ON le.appln_id = a.appln_id
  GROUP BY a.docdb_family_id
)

SELECT
  b.docdb_family_id,
  b.appln_id,

  b.appln_auth, b.appln_nr, b.appln_kind,
  b.appln_filing_date,

  b.publn_auth, b.publn_nr, b.publn_kind, b.publn_date,

  -- Title/abstract + language codes
  COALESCE(te.title_en, ta.title_any) AS appln_title,
  CASE WHEN te.title_en IS NOT NULL THEN 'en' ELSE ta.title_any_lg END AS title_lg,

  COALESCE(ae.abstract_en, aa.abstract_any) AS appln_abstract,
  CASE WHEN ae.abstract_en IS NOT NULL THEN 'en' ELSE aa.abstract_any_lg END AS abstract_lg,

  c.cpc_list,
  i.ipc_list,

  ap.applicant_names,
  ap.applicant_country_codes,

  iv.inventor_names,
  iv.inventor_country_codes,

  -- Priority-or-filing fallback (family-level)
  ep.earliest_priority_date,
  ff.family_earliest_filing_date,
  COALESCE(ep.earliest_priority_date, ff.family_earliest_filing_date) AS earliest_priority_or_filing_date,

  fa.family_publn_auths,
  fa.has_ep, fa.has_wo,

  le.legal_event_count,
  le.first_legal_event_date,
  le.last_legal_event_date,
  le.legal_event_codes

FROM pubs b
JOIN hits h ON h.docdb_family_id = b.docdb_family_id

LEFT JOIN titles_en te ON te.appln_id = b.appln_id
LEFT JOIN titles_any ta ON ta.appln_id = b.appln_id

LEFT JOIN abstracts_en ae ON ae.appln_id = b.appln_id
LEFT JOIN abstracts_any aa ON aa.appln_id = b.appln_id

LEFT JOIN cpc_agg c ON c.appln_id = b.appln_id
LEFT JOIN ipc_agg i ON i.appln_id = b.appln_id

LEFT JOIN applicants ap ON ap.appln_id = b.appln_id
LEFT JOIN inventors iv ON iv.appln_id = b.appln_id

LEFT JOIN earliest_priority ep ON ep.docdb_family_id = b.docdb_family_id
LEFT JOIN family_earliest_filing ff ON ff.docdb_family_id = b.docdb_family_id

LEFT JOIN family_auths fa ON fa.docdb_family_id = b.docdb_family_id
LEFT JOIN legal_events le ON le.docdb_family_id = b.docdb_family_id

ORDER BY b.publn_date;
