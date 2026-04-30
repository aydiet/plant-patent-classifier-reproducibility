# Plant Patent Classifier Appendix D Reproducibility Artifacts

This repository is a public audit companion for Appendix D of the plant-related patent classifier paper. It contains only the source artifacts and safe summary/configuration files listed in the Appendix D reproducibility manifest.

It is not the manuscript repository and it is not a redistributed patent dataset. Manuscript source files, manuscript tables/figures, row-level TIP/PATSTAT exports, patent title/abstract text, translated patent text, model prediction rows, label queues, split manifests, parquet files, and model checkpoints are intentionally excluded.

## Included

- Appendix D workflow scripts and the TIP/PATSTAT SQL export query in `scripts/`.
- The Apple Translation command-line interface source listed in Appendix D, with Swift build outputs removed.
- Safe Appendix D metadata summaries in `metadata/`, including query logs, model comparison reports, model run configurations/metrics, and aggregate external recall results.
- `manifests/excluded_appendix_d_artifacts.csv`, which records listed Appendix D artifacts that are not public, with local sizes and SHA-256 checksums where available.

## Not Included

The excluded artifacts are not included because they are manuscript files, row-level patent data, patent text, translated text, model predictions, family identifiers, large parquet outputs, or local model artifacts. See `DATA_AVAILABILITY.md` and the exclusion manifest for details.

## Reproducing the Workflow

The scripts are preserved in the same relative layout used by the original project, but this repository intentionally includes only the files named by Appendix D. To rerun the workflow, recreate the expected local directories (`data/raw`, `data/derived`, `metadata`, `models`, and `paper` as needed), obtain the required data through the appropriate EPO Technology Intelligence Platform/PATSTAT access route, and run the scripts in the order shown by Appendix D.

The Python environment snapshot used by the working project is recorded in `requirements-freeze.txt`. Some scripts also require external systems or platform-specific tooling, including EPO TIP/PATSTAT access, Apple Translation, Hugging Face model downloads, PyTorch, and DuckDB.

## Citation

If using these materials, cite the associated paper/thesis chapter and this repository version. A formal DOI can be added after a GitHub release or Zenodo archive is created.
