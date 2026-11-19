#Md Arafat Islam 2111293642
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_COLAB = os.path.exists("/content")

if os.environ.get("SEPSIS_DATA_DIR"):
    PROJECT_DIR = os.environ["SEPSIS_DATA_DIR"]
elif IN_COLAB:
    PROJECT_DIR = "/content/drive/MyDrive/early-sepsis-prediction"
else:
    # ./data inside the repo, so it works no matter which folder we run from
    PROJECT_DIR = os.path.join(REPO_DIR, "data")

RAW_DIR = os.path.join(PROJECT_DIR, "raw")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "processed")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")

for folder in [RAW_DIR, PROCESSED_DIR, FIGURES_DIR]:
    os.makedirs(folder, exist_ok=True)

# temp folder for the wget fallback download (local disk, not Drive)
if IN_COLAB:
    TMP_RAW_DIR = "/content/tmp_raw"
else:
    TMP_RAW_DIR = os.path.join(PROJECT_DIR, "tmp_raw")

ZIP_URL = (
    "https://physionet.org/static/published-projects/challenge-2019/"
    "early-prediction-of-sepsis-from-clinical-data-the-physionet-computing-in-cardiology-challenge-2019-1.0.0.zip"
)
FALLBACK_URL = "https://physionet.org/files/challenge-2019/1.0.0/training/"

RAW_ZIP = os.path.join(RAW_DIR, "challenge-2019-training.zip")

RAW_PARQUET = os.path.join(PROCESSED_DIR, "sepsis_raw.parquet")
NODUP_PARQUET = os.path.join(PROCESSED_DIR, "sepsis_nodup.parquet")
DROPPED_PARQUET = os.path.join(PROCESSED_DIR, "sepsis_dropped.parquet")
CLEAN_PARQUET = os.path.join(PROCESSED_DIR, "sepsis_clean.parquet")
SCALED_PARQUET = os.path.join(PROCESSED_DIR, "sepsis_scaled.parquet")
FINAL_PARQUET = os.path.join(PROCESSED_DIR, "sepsis_final.parquet")
FINAL_UNSCALED_PARQUET = os.path.join(PROCESSED_DIR, "sepsis_final_unscaled.parquet")

MISSING_THRESHOLD = 0.90
RANDOM_STATE = 42
PLOT_SAMPLE_PER_CLASS = 5000
SWARM_SAMPLE_PER_CLASS = 250

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp"]

# ---- novel preprocessing pipeline and split ablation ----
# The settings below are only used by novel_preprocessing/ and ablation/.

# 8 vitals + 26 labs = the 34 physiological variables
VITALS_ALL = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
        "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate",
        "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
        "PTT", "WBC", "Fibrinogen", "Platelets"]
MEASURED = VITALS_ALL + LABS

HORIZONS = [6, 12, 24, 48]
K_FOLDS = 5
SPLIT_SEEDS = [42, 43, 44]
# set the environment variable SEPSIS_QUICK_MODE=1 before importing config for a quick run
QUICK_MODE = os.environ.get("SEPSIS_QUICK_MODE", "0") == "1"
QUICK_N_PATIENTS = 4000
TRANSDUCTIVE_P2 = True

# quick-mode checkpoints live in their own folders so they never mix with the full run
NOVEL_DIR = os.path.join(PROCESSED_DIR, "quick") if QUICK_MODE else PROCESSED_DIR
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "quick") if QUICK_MODE else os.path.join(PROJECT_DIR, "results")
NOVEL_FIG_PREFIX = "quick/" if QUICK_MODE else ""
PREDS_DIR = os.path.join(RESULTS_DIR, "preds")
JOBS_DIR = os.path.join(RESULTS_DIR, "jobs")

for folder in [NOVEL_DIR, RESULTS_DIR, PREDS_DIR, JOBS_DIR, os.path.join(FIGURES_DIR, "quick")]:
    os.makedirs(folder, exist_ok=True)

LABELS_PARQUET = os.path.join(NOVEL_DIR, "novel_labels.parquet")
BASE_FEATURES_PARQUET = os.path.join(NOVEL_DIR, "novel_base_features.parquet")
P1_PARQUET = os.path.join(NOVEL_DIR, "novel_p1_observation.parquet")
P3_PARQUET = os.path.join(NOVEL_DIR, "novel_p3_baseline.parquet")
P4_PARQUET = os.path.join(NOVEL_DIR, "novel_p4_scores.parquet")
P5_PARQUET = os.path.join(NOVEL_DIR, "novel_p5_trajectory.parquet")
SPLITS_PARQUET = os.path.join(NOVEL_DIR, "splits.parquet")

EDA_RATES_CSV = os.path.join(RESULTS_DIR, "n00_observation_rates.csv")
EDA_GAPS_CSV = os.path.join(RESULTS_DIR, "n00_measurement_gaps.csv")
EDA_KS_CSV = os.path.join(RESULTS_DIR, "n00_ks_vitals.csv")
TAU_CSV = os.path.join(RESULTS_DIR, "n07_tau_values.csv")
RESULTS_CSV = os.path.join(RESULTS_DIR, "ablation_results.csv")
IMPORTANCE_CSV = os.path.join(RESULTS_DIR, "xgb_importance_novel.csv")
STATS_CI_CSV = os.path.join(RESULTS_DIR, "stats_bootstrap_ci.csv")
STATS_PAIRED_CSV = os.path.join(RESULTS_DIR, "stats_paired_novel_vs_base.csv")
STATS_GAP_CSV = os.path.join(RESULTS_DIR, "stats_generalisation_gap.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary_table.csv")

XGB_PARAMS = {"tree_method": "hist", "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
              "colsample_bytree": 0.8, "objective": "binary:logistic", "eval_metric": "aucpr"}
XGB_MAX_ROUNDS = 2000
XGB_EARLY_STOP = 100
XGB_FIXED_ROUNDS = 500
N_BOOTSTRAP = 1000
P2_REFERENCE_ROWS = 200_000
