import os

import pandas as pd

from data_preprocessing import config, utils
from novel_preprocessing import common


def label_col(h):
    return f"label_{h}"


def make_labels():
    table = common.patient_table()
    print("septic patients:", int(table["ever_septic"].sum()))
    print("excluded by onset_hour <= 6:", int(table["excl_onset_le6"].sum()))
    print("excluded as septic on arrival (first row already labelled):",
          int(table["excl_septic_on_arrival"].sum()))
    patients = common.selected_patients(table)
    if config.QUICK_MODE:
        print(f"QUICK_MODE: using {len(patients)} patients")

    grid = common.load_grid(["SepsisLabel"], patients)
    df = grid[grid["s03_kept"]].drop(columns=["s03_kept"])
    df["onset_hour"] = df["patient_id"].map(table["onset_hour"]).astype("float32")
    df["ever_septic"] = df["patient_id"].map(table["ever_septic"]).astype("int8")

    # rows at or after onset are trivially positive, so they are dropped
    before_onset = df["onset_hour"].isna() | (df["ICULOS"] < df["onset_hour"])
    print(f"dropped {int((~before_onset).sum())} rows at or after onset")
    df = df[before_onset].copy()

    lead = df["onset_hour"] - df["ICULOS"]
    for h in config.HORIZONS:
        # non-septic patients have NaN lead, so the comparison gives 0
        df[label_col(h)] = ((lead > 0) & (lead <= h)).astype("int8")
    df["SepsisLabel"] = df["SepsisLabel"].astype("int8")
    return df


def balance_table(df):
    rows = []
    for h in config.HORIZONS:
        col = label_col(h)
        patient_pos = df.groupby("patient_id")[col].max()
        rows.append({
            "horizon": h,
            "rows": len(df),
            "positive rows": int(df[col].sum()),
            "row %": df[col].mean() * 100,
            "patients": len(patient_pos),
            "positive patients": int(patient_pos.sum()),
            "patient %": patient_pos.mean() * 100,
        })
    return pd.DataFrame(rows).set_index("horizon")


def run(force=False):
    if utils.is_done(config.LABELS_PARQUET) and not force:
        print("loaded checkpoint:", os.path.basename(config.LABELS_PARQUET))
        df = utils.load(config.LABELS_PARQUET)
    else:
        df = make_labels()
        utils.save(df, config.LABELS_PARQUET)

    print("class balance per horizon (row level and patient level):")
    utils.show(balance_table(df).round(2))
    by_hospital = df.groupby("hospital")["patient_id"].nunique()
    print("patients per hospital:", by_hospital.to_dict())
    return df


if __name__ == "__main__":
    run()
