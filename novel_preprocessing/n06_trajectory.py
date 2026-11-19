import os
import time

import numpy as np
import pandas as pd

from data_preprocessing import config, utils
from novel_preprocessing import common, n05_clinical_scores

TRAJECTORY_VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
WINDOWS = [6, 12, 24]


def window_sum(cumsum, pos, w):
    """Sum over the last w rows of the same patient, from a per-patient cumulative sum."""
    out = cumsum.copy()
    has_lag = pos >= w
    rows = np.nonzero(has_lag)[0]
    out[rows] -= cumsum[rows - w]
    return out


def rolling_extremes(values, codes, w):
    """Causal rolling min and max per patient on one global array.

    w empty rows are put in front of every patient, so a window never reaches
    into the previous patient.
    """
    n, k = values.shape
    new_pos = np.arange(n) + w * (codes + 1)
    padded = np.full((n + w * (codes.max() + 1), k), np.nan)
    padded[new_pos] = values
    roll = pd.DataFrame(padded).rolling(w, min_periods=1)
    return roll.min().to_numpy()[new_pos], roll.max().to_numpy()[new_pos]


def features(grid, scores=None):
    """P5 causal rolling mean, std, min, max and least-squares slope."""
    group = grid["patient_id"].to_numpy()
    if scores is None:
        scores = n05_clinical_scores.features(grid)
    series = grid[TRAJECTORY_VITALS].groupby(group, sort=False).ffill()
    series["shock_index"] = scores["shock_index"].to_numpy()
    series["sofa"] = scores["sofa"].to_numpy()
    names = list(series.columns)

    x = series.to_numpy(dtype="float64")
    valid = ~np.isnan(x)
    x0 = np.where(valid, x, 0.0)
    pos = common.positions(grid)
    t = np.where(valid, pos[:, None].astype("float64"), 0.0)
    codes = pd.factorize(group)[0]

    # per-patient cumulative sums start from zero for every patient, so a row only
    # ever depends on its own patient's earlier rows
    def cum(a):
        return pd.DataFrame(a).groupby(codes, sort=False).cumsum().to_numpy()

    sums = {name: cum(a) for name, a in
            [("n", valid.astype("float64")), ("x", x0), ("xx", x0 * x0),
             ("t", t), ("tt", t * t), ("xt", x0 * t)]}

    parts = {}
    for w in WINDOWS:
        s = {name: window_sum(c, pos, w) for name, c in sums.items()}
        n = s["n"]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(n > 0, s["x"] / n, np.nan)
            var = np.where(n > 1, (s["xx"] - s["x"] * s["x"] / n) / (n - 1), np.nan)
            std = np.sqrt(np.clip(var, 0, None))
            # slope = cov(t, x) / var(t) over the valid points of the window
            denom = n * s["tt"] - s["t"] * s["t"]
            slope = np.where((n > 1) & (denom > 0), (n * s["xt"] - s["t"] * s["x"]) / denom, np.nan)
        low, high = rolling_extremes(x, codes, w)
        # float32 right away keeps the peak memory down on the full data
        for i, name in enumerate(names):
            parts[f"{name}_w{w}_mean"] = mean[:, i].astype("float32")
            parts[f"{name}_w{w}_std"] = std[:, i].astype("float32")
            parts[f"{name}_w{w}_min"] = low[:, i].astype("float32")
            parts[f"{name}_w{w}_max"] = high[:, i].astype("float32")
            parts[f"{name}_w{w}_slope"] = slope[:, i].astype("float32")
        del s, mean, var, std, slope, low, high
    return pd.DataFrame(parts, index=grid.index)


def run(force=False):
    labels = utils.load(config.LABELS_PARQUET)
    if utils.is_done(config.P5_PARQUET) and not force:
        print("loaded checkpoint:", os.path.basename(config.P5_PARQUET))
        out = utils.load(config.P5_PARQUET)
    else:
        columns = list(dict.fromkeys(TRAJECTORY_VITALS + n05_clinical_scores.SCORE_INPUTS))
        grid = common.load_grid(columns, labels["patient_id"].unique())
        start = time.time()
        out = features(grid)
        print(f"trajectory features computed on {len(grid)} grid rows in {time.time() - start:.1f} seconds")
        out = common.restrict(out, labels)
        utils.save(out, config.P5_PARQUET)

    print(f"P5 trajectory features: {out.shape[1]} columns")
    utils.show(out[[c for c in out.columns if c.startswith("HR_w6")]].describe().T.round(3))
    return out


if __name__ == "__main__":
    run()
