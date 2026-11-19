import os

import pandas as pd

from data_preprocessing import config, utils
from data_preprocessing.s04_impute_nulls import add_icu_type
from data_preprocessing.s13_encoding import AGE_BINS, AGE_LABELS
from novel_preprocessing import common

# the 14 columns s03 keeps (besides the label and ids)
S03_COLUMNS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "Glucose", "Age", "Gender",
               "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
FFILL_COLUMNS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "Glucose", "Age", "Gender",
                 "HospAdmTime"]
ICU_COLUMNS = ["ICU_MICU", "ICU_SICU", "ICU_Unknown"]
AGE_COLUMNS = [f"Age_{label}" for label in AGE_LABELS]


def features(grid):
    """Stateless BASE features: s03 columns, per-patient forward fill, s13 encodings.

    Values still missing after the forward fill stay NaN; the median comes later
    from training patients only (n07).
    """
    df = add_icu_type(grid[["patient_id"] + S03_COLUMNS].copy())
    # forward fill only, as in s04, so nothing from later hours reaches earlier hours
    df[FFILL_COLUMNS] = df.groupby("patient_id", sort=False)[FFILL_COLUMNS].ffill()

    out = df[FFILL_COLUMNS + ["ICULOS"]].astype("float32")
    for icu in ["MICU", "SICU", "Unknown"]:
        out[f"ICU_{icu}"] = (df["ICU_type"] == icu).astype("float32")
    age_group = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    for label in AGE_LABELS:
        out[f"Age_{label}"] = (age_group == label).astype("float32")
    return out


def run(force=False):
    labels = utils.load(config.LABELS_PARQUET)
    if utils.is_done(config.BASE_FEATURES_PARQUET) and not force:
        print("loaded checkpoint:", os.path.basename(config.BASE_FEATURES_PARQUET))
        out = utils.load(config.BASE_FEATURES_PARQUET)
    else:
        grid = common.load_grid(S03_COLUMNS, labels["patient_id"].unique())
        out = common.restrict(features(grid), labels)
        utils.save(out, config.BASE_FEATURES_PARQUET)

    print(f"BASE features: {out.shape[1]} columns")
    print(", ".join(out.columns))
    missing = out.isna().mean() * 100
    print("\nmissing after forward fill (filled later with the training median):")
    utils.show(missing[missing > 0].round(2).to_frame("missing %"))
    return out


if __name__ == "__main__":
    run()
