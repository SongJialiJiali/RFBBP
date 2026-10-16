# RFBBP

This repository contains:

- `src/rfbbp/`: RFBBP model implementation;
- `case_multivar/`: MIMIC-IV/eICU case-study analyses;
- `experiment/`: simulation experiments;
- `synthetic_example/`: a fully synthetic example dataset for smoke testing.

## Installation

Python 3.13 is used. Install the locked environment from the repository root:

```bash
uv sync --frozen
```

Jupyter can be installed locally if required:

```bash
uv pip install jupyterlab nbclient ipykernel
```

## Data

MIMIC-IV and eICU are credentialed PhysioNet datasets. Users must obtain access independently and comply with the applicable data use agreements. Patient-level source data and derived datasets are not distributed with this repository.

- MIMIC-IV: <https://physionet.org/content/mimiciv/>
- eICU-CRD: <https://physionet.org/content/eicu-crd/2.0/>

Place the harmonized input at:

```text
case_multivar/data/combined_dataset_raw.csv
```

Required fields are:

```text
stay_id, subject_id, hadm_id, database, center_id,
mortality_28d, survival_time, gender, age,
Hemoglobin_hour_24, Lactate_hour_24, Creatinine_hour_24,
SOFA_score_hour_24, Renal_score_hour_24
```

The notebook creates the three analysis datasets under `case_multivar/analysis_final/analysis_datasets/`.

## Case-study analysis

Run the primary RFBBP models:

```bash
uv run bash case_multivar/run_rfbbp.sh primary
```

Run the 70%/30% training–validation analysis:

```bash
uv run python case_multivar/run_train_validate_rfbbp_threshold.py \
  --biomarker all --reference-set final
```

Run prior-sensitivity analyses:

```bash
uv run bash case_multivar/run_rfbbp.sh sensitivity
```

Run the final notebook:

```bash
uv run python -m jupyter nbconvert --execute --to notebook --inplace \
  case_multivar/analysis_final/rfbbp_case_final.ipynb
```

Main RFBBP fits use four chains, 2,000 tuning iterations and 2,000 retained draws per chain. The primary seeds are 3000–3002; the training–validation seed is 20260826. Runtime depends on hardware and may range from hours per model to several days for the complete sensitivity workflow.

Outputs are written under `case_multivar/analysis_final/`. These files are excluded from the public repository by default; only reviewed aggregate tables and publication figures should be released separately.

## Simulation experiments

`experiment/main.py` implements simulation, analysis, evaluation and summarization. `experiment/main.sh` defines the simulation scenarios. These scripts use synthetic data and are separate from the MIMIC-IV/eICU case study.

Use a small smoke test before running the full simulation suite:

```bash
cd experiment
NREPEAT=2 RUN_SCENARIO=3001 NCORES=4 uv run bash main.sh
```

## Synthetic example

`synthetic_example/synthetic_combined_dataset.csv` contains 320 fully synthetic records from four artificial centres. It has the same required columns as the harmonized case-study input and is intended only for pipeline testing; it does not reproduce manuscript estimates.

## Citations

- Johnson AEW, Bulgarelli L, Shen L, et al. MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*. 2023;10:1. <https://doi.org/10.1038/s41597-022-01899-x>
- Pollard TJ, Johnson AEW, Raffa JD, et al. The eICU Collaborative Research Database. *Scientific Data*. 2018;5:180178. <https://doi.org/10.1038/sdata.2018.178>

Please also cite the exact PhysioNet dataset versions used in the analysis.
