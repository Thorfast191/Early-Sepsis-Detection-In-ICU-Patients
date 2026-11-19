import numpy as np
import pandas as pd

from data_preprocessing import config

KEYS = ["patient_id", "hospital", "ICULOS"]


def patient_table():
    """One row per patient: hospital, first ICULOS, first SepsisLabel hour, exclusion flag."""
    df = pd.read_parquet(config.NODUP_PARQUET, columns=["patient_id", "hospital", "ICULOS", "SepsisLabel"])
    g = df.groupby("patient_id")
    table = pd.DataFrame({
        "hospital": g["hospital"].first(),
        "first_hour": g["ICULOS"].min(),
    })
    first_label = df[df["SepsisLabel"] == 1].groupby("patient_id")["ICULOS"].min()
    table["first_label_hour"] = first_label.reindex(table.index)
    # SepsisLabel switches to 1 six hours before clinical onset
    table["onset_hour"] = table["first_label_hour"] + 6
    table["ever_septic"] = table["onset_hour"].notna().astype(int)
    # the spec rule onset_hour <= 6 can never fire (onset_hour >= 7), so we also exclude
    # patients whose very first row is already labelled: their onset is censored, they
    # arrived septic
    table["excl_onset_le6"] = table["onset_hour"] <= 6
    table["excl_septic_on_arrival"] = table["first_label_hour"] == table["first_hour"]
    table["excluded"] = table["excl_onset_le6"] | table["excl_septic_on_arrival"]
    return table


def selected_patients(table=None):
    """Patient ids used by the novel pipeline (all included patients, or a quick sample)."""
    if table is None:
        table = patient_table()
    kept = table[~table["excluded"]]
    if not config.QUICK_MODE:
        return np.array(sorted(kept.index))
    strata = kept["ever_septic"].astype(str) + kept["hospital"].astype(str)
    frac = config.QUICK_N_PATIENTS / len(kept)
    sample = kept.groupby(strata, group_keys=False).sample(frac=frac, random_state=config.RANDOM_STATE)
    return np.array(sorted(sample.index))


def load_grid(columns, patient_ids=None):
    """nodup rows (all ICU hours) of the given patients, sorted by patient_id and ICULOS.

    Rows that s03 drops (every vital missing) stay on the grid as time positions but
    all their measurements are set to NaN, so they add no information. The grid is
    contiguous in ICULOS, so row windows are hour windows.
    """
    measured = [c for c in columns if c in config.MEASURED]
    other = [c for c in columns if c not in config.MEASURED and c not in KEYS]
    read_cols = list(dict.fromkeys(KEYS + config.VITALS + measured + other))
    df = pd.read_parquet(config.NODUP_PARQUET, columns=read_cols)
    if patient_ids is not None:
        df = df[df["patient_id"].isin(set(patient_ids))]
    df = df.sort_values(["patient_id", "ICULOS"], kind="stable")
    dropped_by_s03 = df[config.VITALS].isna().all(axis=1)
    if measured:
        df.loc[dropped_by_s03, measured] = np.nan
    df["s03_kept"] = ~dropped_by_s03
    keep_cols = list(dict.fromkeys(KEYS + measured + other + ["s03_kept"]))
    return df[keep_cols]


def positions(grid):
    """Row position inside each patient (0 for the first hour)."""
    return grid.groupby("patient_id", sort=False).cumcount().to_numpy()


def restrict(features, labels):
    """Keep the labelled rows, in the labels order, and check they line up."""
    out = features.loc[labels.index]
    assert out.index.equals(labels.index), "feature rows do not line up with the labels"
    return out


def to_float32(df):
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype("float32")
    return df
