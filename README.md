# Early Sepsis Prediction in ICU Patients

University ML course project: early prediction of sepsis from hourly ICU data, using the
[PhysioNet/CinC Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/) dataset
(40,336 patients from two hospitals, about 1.5M hourly rows).

This repo currently contains **Part 1: data preprocessing**, split into 13 steps
(`data_preprocessing/s01_*.py` to `s13_*.py`) and run from one notebook.

## Run it in Colab

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

## Run it locally

```
pip install -r requirements.txt
python -m data_preprocessing.s01_read_data
python -m data_preprocessing.s02_drop_duplicates
...
```

Locally the data goes to `./data` (gitignored). Set `SEPSIS_DATA_DIR` to use another folder.
