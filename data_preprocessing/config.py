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
