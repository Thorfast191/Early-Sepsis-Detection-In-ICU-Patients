import os

import numpy as np
import pandas as pd

from data_preprocessing import config, utils
from novel_preprocessing import common

SCORE_INPUTS = ["HR", "SBP", "DBP", "MAP", "Resp", "Temp", "O2Sat", "FiO2", "SaO2", "BUN",
                "Creatinine", "Platelets", "Bilirubin_total", "PaCO2", "WBC"]


def clean_fio2(fio2):
    # FiO2 is a fraction; a few rows are recorded in percent, and some are impossible
    fio2 = fio2.where(fio2 <= 1, fio2 / 100)
    return fio2.where((fio2 >= 0.21) & (fio2 <= 1))


def ratio(a, b):
    return np.where(b > 0, a / b, np.nan)


def band(x, cuts):
    """Score 0..len(cuts): number of cut-offs that x reaches (cuts sorted, x >= cut)."""
    score = np.zeros(len(x))
    for cut in cuts:
        score += (x >= cut)
    return score


def features(grid):
    """P4 hourly clinical scores from forward-filled values."""
    group = grid["patient_id"].to_numpy()
    raw = grid[SCORE_INPUTS].copy()
    raw["FiO2"] = clean_fio2(raw["FiO2"])
    x = {c: s.to_numpy(dtype="float64") for c, s in raw.groupby(group, sort=False).ffill().items()}

    parts = {}
    parts["shock_index"] = ratio(x["HR"], x["SBP"])
    parts["pulse_pressure"] = x["SBP"] - x["DBP"]
    parts["bun_creatinine"] = ratio(x["BUN"], x["Creatinine"])
    parts["sao2_fio2"] = ratio(x["SaO2"], x["FiO2"])
    sf = ratio(x["O2Sat"], x["FiO2"])
    parts["sf_ratio"] = sf

    # partial SOFA; a missing component scores 0 and gets a missing flag
    components = {
        # Rice 2007 S/F cut-offs: <400 =1, <316 =2, <232 =3, <148 =4
        "sofa_resp": 4 - band(sf, [148, 232, 316, 400]),
        "sofa_coag": 4 - band(x["Platelets"], [20, 50, 100, 150]),
        "sofa_liver": band(x["Bilirubin_total"], [1.2, 2, 6, 12]),
        "sofa_cardio": (x["MAP"] < 70).astype("float64"),
        "sofa_renal": band(x["Creatinine"], [1.2, 2, 3.5, 5]),
    }
    inputs = {"sofa_resp": sf, "sofa_coag": x["Platelets"], "sofa_liver": x["Bilirubin_total"],
              "sofa_cardio": x["MAP"], "sofa_renal": x["Creatinine"]}
    sofa = np.zeros(len(grid))
    for name, score in components.items():
        missing = np.isnan(inputs[name])
        score = np.where(missing, 0, score)
        parts[name] = score
        parts[f"{name}_missing"] = missing.astype("float64")
        sofa += score
    parts["sofa"] = sofa
    # Sepsis-3 looks at the rise in SOFA; here the rise from the lowest SOFA so far
    parts["delta_sofa"] = sofa - pd.Series(sofa, index=grid.index).groupby(group, sort=False).cummin().to_numpy()

    parts["qsofa"] = (x["Resp"] >= 22).astype("float64") + (x["SBP"] <= 100)
    parts["sirs"] = (((x["Temp"] > 38) | (x["Temp"] < 36)).astype("float64")
                     + (x["HR"] > 90)
                     + ((x["Resp"] > 20) | (x["PaCO2"] < 32))
                     + ((x["WBC"] > 12) | (x["WBC"] < 4)))
    out = pd.DataFrame(parts, index=grid.index).astype("float32")
    return out


def run(force=False):
    labels = utils.load(config.LABELS_PARQUET)
    if utils.is_done(config.P4_PARQUET) and not force:
        print("loaded checkpoint:", os.path.basename(config.P4_PARQUET))
        out = utils.load(config.P4_PARQUET)
    else:
        grid = common.load_grid(SCORE_INPUTS, labels["patient_id"].unique())
        out = common.restrict(features(grid), labels)
        utils.save(out, config.P4_PARQUET)

    print(f"P4 clinical-score features: {out.shape[1]} columns")
    utils.show(out.describe().T[["mean", "std", "min", "max"]].round(2))
    return out


if __name__ == "__main__":
    run()
