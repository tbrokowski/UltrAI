# ULTR-AI

Basic notes about the current layout of the `UltrAI` codebase for lung ultrasound TB and pathology modeling.

## Top-level scripts
- `trainTB.py` patient-level TB training with attention pooling over sites.
- `train_pathology.py` end-to-end pathology classifier; `train_pathology_ml.py` wraps classical ML baselines.
- `predictTBImage.py` single-patient inference helper.
- `run_all_sensitivity.sh` and `summarize_results.py` automation/analysis utilities.

## Directory guide
- `data/`
  - `images/` ultrasound frames organized by patient and probe site.
  - `labels/` curated CSVs plus helpers for assembling TB/pathology labels.
  - `clinical_data/` merged cohorts and metadata tables.
  - `Splits/` patient ID folds and scripts that build them.
- `data-processing/` standalone scripts for scraping, cleaning, or augmenting raw data.
- `dataloaders/` thin wrappers that batch ultrasound frames + metadata for the training scripts.
- `dataset_loading/` shared dataset objects (single-image loaders, dropout utilities, preprocessing helpers).
- `evaluation/` metric calculations and lightweight evaluation routines used after training.
- `network_architecture/` neural backbones, pooling layers, DeepChest attention blocks, and related training helpers.
- `utilities/` config utilities, logging helpers, and other glue shared between entry points.
- `results_processing_ipynb_files/` exploratory notebooks and notes for downstream analysis/visualizations.

## Quick start
```bash
cd /users/tbrokowski/ULTR-AI-Vid/ULTR-AI/UltrAI
pip install -r requirements.txt
python trainTB.py --help
```
Set `PYTHONPATH` to include this folder (or run `pip install -e .`) so imports such as `network_architecture` and `utilities` resolve cleanly.

## Data expectations
- Ultrasound PNGs named `{patient}_{site}_{frame}.png` in `data/images/`.
- Patient-level label CSVs (`record_id`, `TB Label`, optional metadata) under `data/labels/`.
- Pre-made fold CSVs in `data/Splits/` referenced by the training configs.

## Support
Questions or issues: reach out to the ULTR-AI maintainers (EPFL Intelligent Global Health Lab).
