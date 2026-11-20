import re

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from data_preprocessing import config
from novel_preprocessing import n07_fitted_transforms as n07

DOMAIN_PARAMS = {"tree_method": "hist", "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8,
                 "objective": "binary:logistic", "device": "cpu", "nthread": 4}
DOMAIN_ROUNDS = 100
DOMAIN_FOLDS = 3


def unit_of(col):
    """Feature unit used for screening: one variable's observation or value features,
    one clinical score with its missing flag, or one trajectory series."""
    m = re.match(r"(.+)_w(6|12|24)_(mean|std|min|max|slope)$", col)
    if m:
        return f"{m.group(1)}:traj"
    for suffix in ["_hsince", "_rate", "_ever"]:
        if col.endswith(suffix):
            return f"{col[:-len(suffix)]}:obs"
    for suffix in ["_last", "_dev"]:
        if col.endswith(suffix):
            return f"{col[:-len(suffix)]}:value"
    return f"{col.replace('_missing', '')}:score"


def units(columns):
    out = {}
    for col in columns:
        out.setdefault(unit_of(col), []).append(col)
    return out


def measurement_ratio(p1, hospitals, measured):
    """A/B ratio of the share of rows with a measurement at that hour, per variable."""
    hospitals = np.asarray(hospitals)
    rows = []
    for v in measured:
        observed = (p1[f"{v}_hsince"].to_numpy() == 0) & (p1[f"{v}_ever"].to_numpy() == 1)
        rate_a = observed[hospitals == "A"].mean()
        rate_b = observed[hospitals == "B"].mean()
        ratio = rate_a / rate_b if rate_b > 0 else np.inf
        rows.append({"variable": v, "rate_A": rate_a, "rate_B": rate_b, "ratio_A_B": ratio})
    return pd.DataFrame(rows).set_index("variable")


def domain_auroc(X, hospital_is_a, groups, rng):
    """Out-of-fold AUROC of a small classifier that predicts the hospital from X."""
    if X.shape[1] == 0 or np.nanstd(X, axis=0).max() == 0:
        return 0.5
    pred = np.zeros(len(X))
    folds = GroupKFold(n_splits=DOMAIN_FOLDS)
    for train, test in folds.split(X, hospital_is_a, groups):
        params = dict(DOMAIN_PARAMS, seed=int(rng.integers(1 << 30)))
        booster = xgb.train(params, xgb.DMatrix(X[train], label=hospital_is_a[train]), DOMAIN_ROUNDS)
        pred[test] = booster.predict(xgb.DMatrix(X[test]))
    return roc_auc_score(hospital_is_a, pred)


def balanced_sample(hospitals, n_rows, rng):
    per = n_rows // 2
    rows = []
    for h in ["A", "B"]:
        idx = np.nonzero(hospitals == h)[0]
        rows.append(rng.choice(idx, min(per, len(idx)), replace=False))
    return np.sort(np.concatenate(rows))


def fit_site_screen(train, key, target=None, shared_support=True, screen=True, measured=None):
    """Label-free choice of which novel feature units to keep.

    train and target are dicts with "X" (a row sample of the novel features as the
    model sees them, after P6/P2), "ids" and "hospital" for those rows, and "p1",
    "p1_ids", "p1_hospital" (raw P1 hours-since and ever columns of all rows, for the
    measurement rates). Only feature values and the hospital of each row are used,
    never the sepsis labels. For the external split the training rows come from one
    hospital, so the unlabeled rows of the target hospital are added (unsupervised
    domain adaptation, like the transductive P2). Returns the columns to drop and a
    table of the decisions.
    """
    n07.assert_train_only(train["ids"], key, "site_screen")
    n07.assert_train_only(train["p1_ids"], key, "site_screen_rates")
    measured = config.MEASURED_V2 if measured is None else measured
    parts = [train] if target is None else [train, target]
    X = pd.concat([part["X"] for part in parts])
    hospitals = np.concatenate([np.asarray(part["hospital"]) for part in parts])
    groups = np.concatenate([np.asarray(part["ids"]) for part in parts])
    assert set(np.unique(hospitals)) == {"A", "B"}, "screening needs rows from both hospitals"

    table = []
    drop = set()
    if shared_support:
        p1 = pd.concat([part["p1"] for part in parts])
        p1_hospitals = np.concatenate([np.asarray(part["p1_hospital"]) for part in parts])
        ratio = measurement_ratio(p1, p1_hospitals, measured)
        low, high = config.SHARED_SUPPORT_RATIO
        for v, r in ratio["ratio_A_B"].items():
            outside = not (low <= r <= high)
            if outside:
                drop.update(c for c in [f"{v}_hsince", f"{v}_rate", f"{v}_ever"] if c in X.columns)
            table.append({"unit": f"{v}:obs", "rule": "shared_support", "score": r, "dropped": outside})

    if screen:
        rng = np.random.default_rng(config.RANDOM_STATE)
        rows = balanced_sample(hospitals, config.SCREEN_SAMPLE_ROWS, rng)
        is_a = (hospitals[rows] == "A").astype(int)
        for unit, cols in units([c for c in X.columns if c not in drop]).items():
            values = X[cols].to_numpy(dtype="float32")[rows]
            auc = domain_auroc(values, is_a, groups[rows], rng)
            too_site_specific = auc > config.SCREEN_MAX_DOMAIN_AUROC
            if too_site_specific:
                drop.update(cols)
            table.append({"unit": unit, "rule": "domain_auroc", "score": auc, "dropped": too_site_specific})
    return sorted(drop), pd.DataFrame(table)
