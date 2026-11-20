import os

import pandas as pd

from data_preprocessing import config, utils
from novel_preprocessing import (common, n03_observation_process, n04_patient_baseline,
                                 n05_clinical_scores)
from novel_preprocessing import n07_fitted_transforms as n07


def fix_grid(grid):
    """Split a variable into two by value: below the cut-off is a different measurement.

    Calcium below 2.0 is ionized calcium in mmol/L (hospital B), not total calcium in
    mg/dL. SaO2 below 80 is most likely a venous sample (hospital A). No unit conversion
    is applied; the low values just become their own variable.
    """
    grid = grid.copy()
    for old, (new, cut) in config.VARIABLE_SPLITS.items():
        low = grid[old] < cut
        grid[new] = grid[old].where(low)
        grid[old] = grid[old].where(~low)
    return grid


def features(grid):
    """v2 stateless features that change with the data fixes: P1, P3 and P4."""
    fixed = fix_grid(grid)
    return (n03_observation_process.features(fixed, config.MEASURED_V2),
            n04_patient_baseline.features(fixed, config.LABS_V2),
            n05_clinical_scores.features(fixed))


def moved_table(grid):
    rows = []
    for old, (new, cut) in config.VARIABLE_SPLITS.items():
        for hospital in ["A", "B"]:
            values = grid.loc[grid["hospital"] == hospital, old].dropna()
            rows.append({"variable": old, "new variable": new, "cut-off": cut, "hospital": hospital,
                         "measurements": len(values), "moved": int((values < cut).sum()),
                         "moved %": (values < cut).mean() * 100 if len(values) else 0})
    return pd.DataFrame(rows)


def tau_table(labels, p1_v1, p1_v2):
    key = ("S1", config.SPLIT_SEEDS[0], 0)
    train = labels["patient_id"].isin(n07.train_ids(key)).to_numpy()
    ids, hours = labels.loc[train, "patient_id"], labels.loc[train, "ICULOS"]
    v1 = n07.fit_decay(p1_v1[train], ids, hours, key)["tau"]
    v2_first = n07.fit_decay(p1_v2[train], ids, hours, key, config.MEASURED_V2, "first")["tau"]
    v2_exp = n07.fit_decay(p1_v2[train], ids, hours, key, config.MEASURED_V2, "exp")["tau"]
    return pd.DataFrame({"v1 first crossing": v1, "v2 first crossing": v2_first,
                         "v2 exponential fit": v2_exp.round(1)})


def run(force=False):
    labels = utils.load(config.LABELS_PARQUET)
    paths = [config.P1_V2_PARQUET, config.P3_V2_PARQUET, config.P4_V2_PARQUET]
    columns = list(dict.fromkeys(config.MEASURED + n05_clinical_scores.SCORE_INPUTS))
    grid = common.load_grid(columns, labels["patient_id"].unique())
    print("values moved to a separate variable:")
    utils.show(moved_table(grid[grid["s03_kept"]]).round(2))

    if all(utils.is_done(p) for p in paths) and not force:
        for p in paths:
            print("loaded checkpoint:", os.path.basename(p))
        p1, p3, p4 = (utils.load(p) for p in paths)
    else:
        p1, p3, p4 = (common.restrict(f, labels) for f in features(grid))
        for frame, path in zip([p1, p3, p4], paths):
            utils.save(frame, path)
    del grid

    from ablation import a01_make_splits
    a01_make_splits.run()
    tau = tau_table(labels, utils.load(config.P1_PARQUET), p1)
    tau.to_csv(config.TAU_V2_CSV)
    print("\ntau_v (hours) on S1 seed 42 training patients, v1 rule vs v2 exponential fit:")
    utils.show(tau)
    return p1, p3, p4


if __name__ == "__main__":
    run()
