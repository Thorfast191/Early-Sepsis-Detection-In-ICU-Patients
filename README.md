# Early Sepsis Prediction in ICU Patients

CSE445 ML course project: early prediction of sepsis from hourly ICU data, using the
[PhysioNet/CinC Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/) dataset contains 40,336 patients data from two hospitals, about 1.5M hourly rows.

This repo currently contains **Part 1: data preprocessing**, split into 13 steps
(`data_preprocessing/s01_*.py` to `s13_*.py`) and run from one notebook.

## For Group Mates

1. Clone the repo
   2\. Navigate to the `/data_preprocessing` directory and review all files and their comments. Each student who contributed to a file must add their full name and student ID in the comments.

## How to run it in colab??

1. Open `notebooks/01_data_preprocessing.ipynb` in Google Colab.
2. Run the setup cells. They mount Google Drive, clone (or `git pull`) this repo into
   `/content`, and install `requirements.txt`.
3. Run the step sections in order.

All data lives in Drive under `MyDrive/early-sepsis-prediction/`:

```
raw/         dataset zip (downloaded once)
processed/   parquet checkpoints of every data step
figures/     saved plots and the IQR outlier table
```

Every data step (1, 2, 3, 4, 12, 13) saves a parquet checkpoint. In a new session a finished
step loads its checkpoint instead of recomputing, but still shows its tables and plots.
Set `FORCE = True` at the top of the notebook to redo all data steps.

The first run downloads the dataset. PhysioNet's single-zip link currently returns 404, so the
code falls back to downloading the 40,336 `.psv` files in parallel from the PhysioNet file
listing and writes them straight into one zip in Drive (`raw/`). Later sessions read that zip
and never download again.

## Part 2 and 3: novel preprocessing and split ablation

`novel_preprocessing/` builds a leakage-free BASE feature set and novel features (observation process,
patient baseline, clinical scores, trajectories, time-decay imputation, hospital-relative encoding).
`ablation/` compares BASE and NOVEL under train/val/test, train/test, 5-fold CV and cross-hospital
splits, with XGBoost and logistic regression, for horizons of 6, 12, 24 and 48 hours.

1. Run notebook 01 at least up to step 2 (`sepsis_nodup.parquet`).
2. Run `notebooks/02_novel_preprocessing.ipynb` (features, a few minutes, CPU is enough).
3. Run `notebooks/03_split_ablation.ipynb` on a GPU runtime. Every trained model writes its own
   result file, so after a disconnect just run the cells again.

Results go to `results/` in Drive (`ablation_results.csv` in long format, statistics, summary table).
Set `QUICK_MODE = True` in the notebooks (or `SEPSIS_QUICK_MODE=1` locally) to run everything on a
sample of 4,000 patients with separate checkpoint folders. The checks in `tests/` (causality, leakage,
identical evaluation sets) run with `python -m tests.test_causality` and so on.

## Run it locally

```
pip install -r requirements.txt
python -m data_preprocessing.s01_read_data
python -m data_preprocessing.s02_drop_duplicates
...
```

Locally the data goes to `./data` (gitignored). Set `SEPSIS_DATA_DIR` to use another folder.
