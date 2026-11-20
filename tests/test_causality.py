import numpy as np
import pandas as pd

from data_preprocessing import config, utils
from novel_preprocessing import (common, n02_base_features, n03_observation_process, n04_patient_baseline,
                                 n05_clinical_scores, n06_trajectory, n08_data_fixes)

N_PATIENTS = 50


def all_features(grid):
    return pd.concat([
        n02_base_features.features(grid),
        n03_observation_process.features(grid),
        n04_patient_baseline.features(grid),
        n05_clinical_scores.features(grid),
        n06_trajectory.features(grid),
    ] + [f.add_suffix("__v2") for f in n08_data_fixes.features(grid)], axis=1)


def run():
    rng = np.random.default_rng(0)
    labels = utils.load(config.LABELS_PARQUET)
    patients = rng.choice(labels["patient_id"].unique(), N_PATIENTS, replace=False)
    columns = list(dict.fromkeys(n02_base_features.S03_COLUMNS + config.MEASURED))
    grid = common.load_grid(columns, patients)
    full = all_features(grid)

    # cut every patient at a random hour and recompute from the cut data only
    pos = common.positions(grid)
    cut = grid["patient_id"].map({p: rng.integers(1, n + 1) for p, n in
                                  grid.groupby("patient_id")["ICULOS"].size().items()}).to_numpy()
    truncated = grid[pos < cut]
    assert len(truncated) < len(grid)
    part = all_features(truncated)
    expected = full.loc[part.index]
    pd.testing.assert_frame_equal(part, expected, check_exact=True)
    print(f"causality OK: {N_PATIENTS} patients, {len(truncated)} of {len(grid)} rows kept after random cuts, "
          f"{part.shape[1]} stateless features identical (exact)")


if __name__ == "__main__":
    run()
