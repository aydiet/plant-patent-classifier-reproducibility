# Data Availability

This repository does not redistribute patent data from the EPO Technology Intelligence Platform, PATSTAT, external validation registries, or derived row-level patent corpora.

The public package is limited to Appendix D workflow code and safe audit metadata. Excluded files include raw and derived parquet exports, patent title/abstract fields, machine-translated text, annotation queues, split manifests, family-level identifiers, model prediction rows, and model checkpoints.

Users who want to rerun the workflow need to regenerate the excluded artifacts in their own controlled environment using appropriate access to the underlying patent-data services. The expected local paths and checksums for excluded Appendix D artifacts are listed in `manifests/excluded_appendix_d_artifacts.csv`.

The exclusion of data files is intentional. Public patent records may be searchable individually, but structured exports and derived row-level corpora can still be subject to platform/database terms and should not be treated as freely redistributable.
