import os

import pandas as pd

from data_preprocessing import config, utils
from novel_preprocessing import common

BASELINE_VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
BASELINE_HOURS = 6


def first_observed(values, group):
    """Value of the first measurement, carried forward (NaN before it)."""
    observed = values.notna()
    first = observed & (observed.groupby(group, sort=False).cumsum() == 1)
    return values.where(first).groupby(group, sort=False).ffill()


def features(grid, labs=None):
    """P3 deviation from the patient's own baseline."""
    labs = config.LABS if labs is None else labs
    group = grid["patient_id"].to_numpy()
    pos = common.positions(grid)
    last = grid[BASELINE_VITALS + labs].groupby(group, sort=False).ffill()
    parts = {}

    # vitals: expanding median over the first 6 hours, then frozen
    in_window = pos < BASELINE_HOURS
    window = grid.loc[in_window, ["patient_id"] + BASELINE_VITALS]
    window_median = (window.groupby("patient_id", sort=False)[BASELINE_VITALS]
                     .expanding(min_periods=1).median()
                     .reset_index(level=0, drop=True))
    baseline = pd.DataFrame(index=grid.index, columns=BASELINE_VITALS, dtype="float64")
    baseline.loc[window_median.index, BASELINE_VITALS] = window_median[BASELINE_VITALS].to_numpy()
    # after hour 6 nothing new is added, so the last window value is carried forward
    baseline = baseline.groupby(group, sort=False).ffill()
    # never measured in the first 6 hours: the first later measurement becomes the baseline
    baseline = baseline.fillna(first_observed(grid[BASELINE_VITALS], group))
    for v in BASELINE_VITALS:
        parts[f"{v}_dev"] = (last[v] - baseline[v]).to_numpy(dtype="float32")

    labs_first = first_observed(grid[labs], group)
    for v in labs:
        parts[f"{v}_dev"] = (last[v] - labs_first[v]).to_numpy(dtype="float32")
    return pd.DataFrame(parts, index=grid.index)


def run(force=False):
    labels = utils.load(config.LABELS_PARQUET)
    if utils.is_done(config.P3_PARQUET) and not force:
        print("loaded checkpoint:", os.path.basename(config.P3_PARQUET))
        out = utils.load(config.P3_PARQUET)
    else:
        grid = common.load_grid(config.MEASURED, labels["patient_id"].unique())
        out = common.restrict(features(grid), labels)
        utils.save(out, config.P3_PARQUET)

    print(f"P3 baseline-deviation features: {out.shape[1]} columns")
    utils.show(out[[f"{v}_dev" for v in BASELINE_VITALS]].describe().T.round(2))
    return out


if __name__ == "__main__":
    run()
