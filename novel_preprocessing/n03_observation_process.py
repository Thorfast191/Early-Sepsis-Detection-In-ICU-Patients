import os

import numpy as np
import pandas as pd

from data_preprocessing import config, utils
from novel_preprocessing import common

# EtCO2 is almost never measured, so only its flag and timing are kept
NO_VALUE = ["EtCO2"]


def features(grid):
    """P1 observation process per variable: last value, hours since, ever observed, rate."""
    variables = config.MEASURED
    values = grid[variables]
    observed = values.notna()
    group = grid["patient_id"].to_numpy()
    hours = grid["ICULOS"].to_numpy(dtype="float64")

    counts = observed.groupby(group, sort=False).cumsum().to_numpy(dtype="float64")
    obs_time = pd.DataFrame(np.where(observed.to_numpy(), hours[:, None], np.nan),
                            index=grid.index, columns=variables)
    last_time = obs_time.groupby(group, sort=False).ffill().to_numpy()
    last_value = values.groupby(group, sort=False).ffill()

    ever = counts > 0
    hours_since = np.where(ever, hours[:, None] - last_time, hours[:, None])
    rate = counts / hours[:, None]

    parts = {}
    for i, v in enumerate(variables):
        if v not in NO_VALUE:
            parts[f"{v}_last"] = last_value[v].to_numpy(dtype="float32")
        parts[f"{v}_hsince"] = hours_since[:, i].astype("float32")
        parts[f"{v}_ever"] = ever[:, i].astype("float32")
        parts[f"{v}_rate"] = rate[:, i].astype("float32")
    return pd.DataFrame(parts, index=grid.index)


def run(force=False):
    labels = utils.load(config.LABELS_PARQUET)
    if utils.is_done(config.P1_PARQUET) and not force:
        print("loaded checkpoint:", os.path.basename(config.P1_PARQUET))
        out = utils.load(config.P1_PARQUET)
    else:
        grid = common.load_grid(config.MEASURED, labels["patient_id"].unique())
        out = common.restrict(features(grid), labels)
        utils.save(out, config.P1_PARQUET)

    print(f"P1 observation-process features: {out.shape[1]} columns")
    rates = out[[f"{v}_rate" for v in config.MEASURED]].mean().round(3)
    print("mean measurement rate per hour so far (first 10 variables):")
    utils.show(rates.head(10).to_frame("rate"))
    return out


if __name__ == "__main__":
    run()
