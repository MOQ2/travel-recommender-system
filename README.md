# Travel Recommender System

This repository contains the core data pipeline, analysis scripts, and notebooks for a multimodal travel recommendation/classification project.

The project focuses on combining text and image-derived signals to analyze destination preferences and generate model/report artifacts.




## What Is Included

- Core Python scripts in `src/` and project root
- Processed datasets in `data/processed/`
- Analysis notebooks in `notebooks/`
- Generated metrics/figures in `results/`

## What Is Intentionally Excluded

To keep the repository lightweight and easy to clone, heavy/generated assets are ignored:

- `models/` (trained `.joblib` files)
- `data/raw/` and `data/features/`
- cache directories and temporary reports

See `.gitignore` for full details.

## Repository Structure

```
travel-recommender-system/
|-- src/
|   |-- merge_csvs_and_skip_bad.py
|   |-- image_qc.py
|   `-- view_csv.py
|-- notebooks/
|-- data/
|   `-- processed/
|-- results/
|-- generate_report_figures.py
|-- analyze_data.py
|-- check_data.py
|-- fix_source_file.py
`-- requirements.txt
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Typical Workflow

Run from the repository root:

1. Merge and clean raw CSV submissions:

```bash
python src/merge_csvs_and_skip_bad.py
```

2. Run image quality control and RGB normalization:

```bash
python src/image_qc.py
```

3. Validate/enrich dataset for EDA:

```bash
python notebooks/run_eda_analysis.py
```

4. Generate grouped report figures:

```bash
python generate_report_figures.py
```

## Notes

- `src/merge_csvs_and_skip_bad.py` expects raw submissions under `data/raw/compressed-attachments/`.
- Processed CSVs are written under `data/processed/`.
- Results are written under `results/`.
