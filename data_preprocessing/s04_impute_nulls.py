import os

import pandas as pd

from data_preprocessing import config, utils


def add_icu_type(df):
    df["ICU_type"] = "Unknown"
    if "Unit2" in df.columns:
        df.loc[df["Unit2"] == 1, "ICU_type"] = "SICU"
    if "Unit1" in df.columns:
        df.loc[df["Unit1"] == 1, "ICU_type"] = "MICU"
    df = df.drop(columns=utils.existing(["Unit1", "Unit2"], df))
    return df


def impute(df):
    df = add_icu_type(df.copy())

    df = df.sort_values(["patient_id", "ICULOS"]).reset_index(drop=True)

    numeric_cols = []
    for col in df.columns:
        if col != "SepsisLabel" and pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    # forward fill only: a value measured at hour t is carried to later hours,
    # never back to earlier ones, so no future information leaks into the past
    df[numeric_cols] = df.groupby("patient_id")[numeric_cols].ffill()

    # median, because most of these columns are skewed
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in df.columns:
        if col not in numeric_cols and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


BEFORE_COLS = config.VITALS + ["Glucose", "Unit1", "Unit2"]
AFTER_COLS = config.VITALS + ["Glucose", "ICU_type"]
YELLOW = "#fff3b0"
GREEN = "#c8f7c5"


def before_rows(df):
    patient_id = utils.demo_patient()
    return utils.demo_rows(df, patient_id, n=12, columns=utils.existing(BEFORE_COLS, df))


def before():
    df = utils.load(config.DROPPED_PARQUET)
    rows = before_rows(df)
    utils.heading(f"Before: demo patient {utils.demo_patient()}, first 12 hours (missing cells in yellow)")
    utils.show_highlighted(rows, rows.isna(), YELLOW)


def show_after(before_df, after_df):
    old = before_rows(before_df)
    new = utils.demo_rows(after_df, utils.demo_patient(), hours=list(old["ICULOS"]),
                          columns=utils.existing(AFTER_COLS, after_df))
    assert list(old["ICULOS"]) == list(new["ICULOS"]), "demo hours do not line up"

    # green marks the cells that were missing before and are filled now
    was_missing = pd.DataFrame(False, index=new.index, columns=new.columns)
    for col in new.columns:
        if col in old.columns:
            was_missing[col] = old[col].isna().values
    utils.heading("After: same hours (cells that were missing in green)")
    utils.show_highlighted(new, was_missing, GREEN)


def run(force=False):
    before = utils.load(config.DROPPED_PARQUET)

    if utils.is_done(config.CLEAN_PARQUET) and not force:
        print("loaded checkpoint:", os.path.basename(config.CLEAN_PARQUET))
        after = utils.load(config.CLEAN_PARQUET)
    else:
        after = impute(before)
        utils.save(after, config.CLEAN_PARQUET)

    print("ICU_type counts (built from Unit1 / Unit2):")
    icu_counts = after["ICU_type"].value_counts()
    utils.show(pd.DataFrame({"rows": icu_counts,
                             "percent": (icu_counts / len(after) * 100).round(1)}))
    unknown_share = (after["ICU_type"] == "Unknown").mean() * 100
    print(f"Unknown share: {unknown_share:.1f}% (kept as its own category instead of mode filling)")

    missing_before = before.isna().sum()
    missing_after = after.isna().sum()
    table = pd.DataFrame({"missing before": missing_before, "missing after": missing_after})
    table = table.fillna(0).astype(int).sort_values("missing before", ascending=False)
    print("\nmissing values before vs after imputation:")
    utils.show(table)

    assert after.isna().sum().sum() == 0, "there are still nulls after imputation"
    print("no missing values left, shape:", after.shape)

    show_after(before, after)
    return after


if __name__ == "__main__":
    run()
