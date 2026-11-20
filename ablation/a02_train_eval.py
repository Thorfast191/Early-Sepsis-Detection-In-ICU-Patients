import glob
import os
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ablation import a01_make_splits
from data_preprocessing import config, utils
from novel_preprocessing import n07_fitted_transforms as n07
from novel_preprocessing import n09_site_screening as n09
from novel_preprocessing.n01_labels import label_col

MODELS = ["xgb", "logreg"]
STRATEGIES = ["S1", "S2", "S3", "S4"]
# the other P2 variant, run on the external split only
ALT_P2_ARM = "NOVEL_inductiveP2" if config.TRANSDUCTIVE_P2 else "NOVEL_transductiveP2"
TARGET_SPECIFICITY = 0.90

# v2 arms (config.RUN_V2), trained on config.V2_SPLITS only. Each is NOVEL with options;
# BASE_noICULOS is BASE without ICULOS.
V2_FIXED = {"v2_data": True, "tau": "exp"}
V2_ARMS = {
    "NOVEL_fix": V2_FIXED,
    "NOVEL_fix_share": {**V2_FIXED, "shared_support": True},
    "NOVEL_fix_screen": {**V2_FIXED, "screen": True},
    "NOVEL_fix_share_screen": {**V2_FIXED, "shared_support": True, "screen": True},
    "NOVEL_minus_P1value": {"drop_suffixes": ["_last"]},
    "NOVEL_minus_P1hsince": {"drop_suffixes": ["_hsince"]},
    "NOVEL_minus_P1rate": {"drop_suffixes": ["_rate"]},
    "NOVEL_minus_P1ever": {"drop_blocks": ["ever"]},
    "NOVEL_minus_P2": {"no_p2": True},
    "NOVEL_minus_P3": {"drop_blocks": ["p3"]},
    "NOVEL_minus_P4": {"drop_blocks": ["p4"]},
    "NOVEL_minus_P5": {"drop_blocks": ["p5"]},
    "NOVEL_minus_P6": {"no_p6": True},
    "NOVEL_noICULOS": {"drop_base": ["ICULOS"]},
}


# ---- official PhysioNet 2019 utility (Reyna et al. 2020, evaluate_sepsis_score.py) ----

def compute_prediction_utility(labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3.0,
                               max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0):
    """Utility of one patient's binary predictions, as in the official script."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float("inf")
    n = len(labels)
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            elif not is_septic and not predictions[t]:
                u[t] = u_tn
    return np.sum(u)


def normalized_utility(frame, dt_early=-12, dt_optimal=-6, dt_late=3):
    """Normalised cohort utility; frame has patient_id, ICULOS, SepsisLabel and alarm."""
    observed = best = inaction = 0.0
    for _, rows in frame.sort_values(["patient_id", "ICULOS"]).groupby("patient_id", sort=False):
        labels = rows["SepsisLabel"].to_numpy()
        n = len(labels)
        best_pred = np.zeros(n)
        if labels.any():
            t_sepsis = np.argmax(labels) - dt_optimal
            best_pred[max(0, t_sepsis + dt_early): min(t_sepsis + dt_late + 1, n)] = 1
        observed += compute_prediction_utility(labels, rows["alarm"].to_numpy())
        best += compute_prediction_utility(labels, best_pred)
        inaction += compute_prediction_utility(labels, np.zeros(n))
    if best == inaction:
        return np.nan
    return (observed - inaction) / (best - inaction)


# ---- metrics ----

def threshold_at_specificity(y, p, specificity=TARGET_SPECIFICITY):
    negatives = p[y == 0]
    return float(np.quantile(negatives, specificity)) if len(negatives) else 0.5


def evaluate(test, threshold, horizon):
    """Row-level, alarm-level and patient-level metrics of one test set."""
    y, p = test["y"].to_numpy(), test["p"].to_numpy()
    alarm = p > threshold
    tp = int((alarm & (y == 1)).sum())
    fp = int((alarm & (y == 0)).sum())
    fn = int((~alarm & (y == 1)).sum())
    tn = int((~alarm & (y == 0)).sum())
    m = {
        "auroc": roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
        "auprc": average_precision_score(y, p) if y.sum() > 0 else np.nan,
        "brier": brier_score_loss(y, p),
        "threshold": threshold,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "false_alarms_per_patient_day": fp / (len(test) / 24),
        "test_rows": len(test),
        "test_positive_rows": int(y.sum()),
    }

    frame = test.assign(alarm=alarm)
    septic = frame[frame["onset_hour"].notna()]
    first_alarm = septic[septic["alarm"]].groupby("patient_id")["ICULOS"].min()
    onset = septic.groupby("patient_id")["onset_hour"].first()
    lead = onset.loc[first_alarm.index] - first_alarm
    m["septic_patients"] = len(onset)
    m["patient_detection_rate"] = len(first_alarm) / len(onset) if len(onset) else np.nan
    m["median_lead_time_h"] = float(lead.median()) if len(lead) else np.nan
    non_septic = frame[frame["onset_hour"].isna()]
    m["non_septic_patients_with_alarm"] = non_septic.groupby("patient_id")["alarm"].any().mean()
    if horizon == 6:
        m["utility"] = normalized_utility(frame[["patient_id", "ICULOS", "SepsisLabel", "alarm"]])
    return m


# ---- data and features per split ----

def load_data():
    labels = utils.load(config.LABELS_PARQUET)
    parts = {"base": config.BASE_FEATURES_PARQUET, "p1": config.P1_PARQUET, "p3": config.P3_PARQUET,
             "p4": config.P4_PARQUET, "p5": config.P5_PARQUET}
    if config.RUN_V2:
        parts.update({"p1_v2": config.P1_V2_PARQUET, "p3_v2": config.P3_V2_PARQUET,
                      "p4_v2": config.P4_V2_PARQUET})
    data = {"labels": labels}
    for name, path in parts.items():
        data[name] = utils.load(path)
        assert data[name].index.equals(labels.index), f"{name} rows do not line up with the labels"
    return data


def split_roles(labels, key):
    strategy, seed, fold = key
    s = n07.load_splits()
    s = s[(s["strategy"] == strategy) & (s["seed"] == seed) & (s["fold"] == fold)]
    return labels["patient_id"].map(s.set_index("patient_id")["role"]).to_numpy()


def fill_and_scale(X, train, ids, key):
    """Training-median fill and StandardScaler, fitted on training rows, applied in place."""
    medians = n07.fit_medians(X[train], ids[train], key)
    n07.apply_medians(X, medians)
    scaler = n07.fit_scaler(X[train], ids[train], key)
    n07.apply_scaler(X, scaler)


def base_matrix(data, roles, key):
    """BASE features as one float32 array (rows = labels rows) and their names."""
    ids = data["labels"]["patient_id"].to_numpy()
    X = data["base"].to_numpy(dtype="float32", copy=True)
    fill_and_scale(X, roles == "train", ids, key)
    return X, list(data["base"].columns)


def drop_base_columns(base, columns):
    X, cols = base
    keep = [i for i, c in enumerate(cols) if c not in columns]
    return X[:, keep], [cols[i] for i in keep]


def screening_inputs(blocks, p1, rows, ids, hospitals, rng):
    """Row sample of the novel model inputs and the raw P1 rate columns, for n09."""
    pick = rows.nonzero()[0]
    if len(pick) > 2 * config.SCREEN_SAMPLE_ROWS:
        pick = np.sort(rng.choice(pick, 2 * config.SCREEN_SAMPLE_ROWS, replace=False))
    X = pd.concat([b.iloc[pick] for b in blocks], axis=1)
    rate_cols = [c for c in p1.columns if c.endswith("_hsince") or c.endswith("_ever")]
    return {"X": X, "ids": ids[pick], "hospital": hospitals[pick], "p1": p1.loc[rows, rate_cols],
            "p1_ids": ids[rows], "p1_hospital": hospitals[rows]}


def novel_matrix(data, roles, key, base, transductive, opts=None, arm="NOVEL"):
    """BASE + novel features as one float32 array; base is the output of base_matrix.

    opts (v2 arms only, None = v1): v2_data, tau, no_p6, no_p2, drop_blocks, drop_suffixes,
    drop_base, shared_support, screen.
    """
    opts = opts or {}
    labels = data["labels"]
    ids = labels["patient_id"].to_numpy()
    hospitals = labels["hospital"].astype(str).to_numpy()
    train = roles == "train"
    v2 = opts.get("v2_data", False)
    p1 = data["p1_v2"] if v2 else data["p1"]
    measured = config.MEASURED_V2 if v2 else config.MEASURED

    if opts.get("no_p6"):
        last = p1[[f"{v}_last" for v in n07.value_variables(measured)]]
    else:
        decay = n07.fit_decay(p1[train], ids[train], labels["ICULOS"].to_numpy()[train], key,
                              measured, opts.get("tau", "first"))
        last = n07.apply_decay(p1, decay)
    hs_rate = p1[[c for c in n07.p2_columns(measured) if not c.endswith("_last")]]
    p2_in = pd.concat([last, hs_rate], axis=1)
    del last

    if opts.get("no_p2"):
        p2_out = p2_in
    else:
        refs = n07.fit_p2(p2_in[train], ids[train], hospitals[train], key)
        ref_hospital = hospitals
        if key[0] == "S4":
            source, target = a01_make_splits.S4_DIRECTIONS[key[2]]
            test = roles == "test"
            if transductive:
                # unsupervised adaptation: target hospital features only, no labels
                refs.update(n07.fit_p2_unlabeled(p2_in[test], target))
            else:
                # the target hospital is encoded with the training hospital's reference
                ref_hospital = np.where(test, source, hospitals)
        keys = n07.row_keys(ids, labels["ICULOS"].to_numpy())
        p2_out = n07.apply_p2(p2_in, ref_hospital, refs, keys)
    del p2_in

    named = {"p2": p2_out, "ever": p1[[f"{v}_ever" for v in measured]],
             "p3": data["p3_v2"] if v2 else data["p3"], "p4": data["p4_v2"] if v2 else data["p4"],
             "p5": data["p5"]}
    for name in opts.get("drop_blocks", []):
        named.pop(name)
    blocks = []
    for block in named.values():
        drop = [c for c in block.columns if any(c.endswith(sfx) for sfx in opts.get("drop_suffixes", []))]
        blocks.append(block.drop(columns=drop) if drop else block)

    if opts.get("shared_support") or opts.get("screen"):
        rng = np.random.default_rng(config.RANDOM_STATE)
        train_part = screening_inputs(blocks, p1, train, ids, hospitals, rng)
        target_part = screening_inputs(blocks, p1, roles == "test", ids, hospitals, rng) \
            if key[0] == "S4" else None
        drop, table = n09.fit_site_screen(train_part, key, target_part, opts.get("shared_support", False),
                                          opts.get("screen", False), measured)
        save_screen_table(table, arm, key)
        print(f"{arm} {key}: site screening drops {len(drop)} columns")
        blocks = [b.drop(columns=[c for c in b.columns if c in set(drop)]) for b in blocks]

    if opts.get("drop_base"):
        base = drop_base_columns(base, opts["drop_base"])
    base_X, base_cols = base
    columns = base_cols + [c for b in blocks for c in b.columns]
    # one array for everything, filled block by block, so no extra full-size copies
    X = np.empty((len(labels), len(columns)), dtype="float32")
    X[:, :len(base_cols)] = base_X
    start = len(base_cols)
    for block in blocks:
        X[:, start:start + block.shape[1]] = block.to_numpy(dtype="float32")
        start += block.shape[1]
    del blocks, p2_out, named
    fill_and_scale(X[:, len(base_cols):], train, ids, key)
    return X, columns


def save_screen_table(table, arm, key):
    strategy, seed, fold = key
    table = table.assign(arm=arm, strategy=strategy, seed=seed, fold=fold)
    old = pd.read_csv(config.SCREEN_CSV) if utils.is_done(config.SCREEN_CSV) else pd.DataFrame()
    if len(old):
        old = old[~((old["arm"] == arm) & (old["strategy"] == strategy) & (old["seed"] == seed)
                    & (old["fold"] == fold))]
    pd.concat([old, table], ignore_index=True).to_csv(config.SCREEN_CSV, index=False)


# ---- models ----

_DEVICE = None


def xgb_device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = "cpu"
        if xgb.build_info().get("USE_CUDA"):
            try:
                xgb.train({"device": "cuda", "tree_method": "hist"},
                          xgb.DMatrix(np.array([[0.0], [1.0]]), label=[0, 1]), 1)
                _DEVICE = "cuda"
            except xgb.core.XGBoostError:
                pass
    return _DEVICE


def train_xgb(dtrain, dval, y_train):
    params = dict(config.XGB_PARAMS)
    params["device"] = xgb_device()
    params["scale_pos_weight"] = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    params["seed"] = config.RANDOM_STATE
    if dval is None:
        booster = xgb.train(params, dtrain, num_boost_round=config.XGB_FIXED_ROUNDS)
        return booster, config.XGB_FIXED_ROUNDS
    booster = xgb.train(params, dtrain, num_boost_round=config.XGB_MAX_ROUNDS,
                        evals=[(dval, "val")], early_stopping_rounds=config.XGB_EARLY_STOP,
                        verbose_eval=False)
    return booster, booster.best_iteration + 1


def xgb_predict(booster, dmatrix, rounds):
    return booster.predict(dmatrix, iteration_range=(0, rounds))


def train_logreg(X, y):
    model = LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=300)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X, y)
    return model


# ---- jobs ----

def job_name(arm, key, horizon, model):
    strategy, seed, fold = key
    return f"{arm}__{strategy}__{seed}__{fold}__H{horizon}__{model}"


def job_done(name):
    return os.path.exists(os.path.join(config.JOBS_DIR, name + ".csv"))


def arms_for(key):
    arms = ["BASE", "NOVEL", ALT_P2_ARM] if key[0] == "S4" else ["BASE", "NOVEL"]
    if config.RUN_V2 and tuple(key) in [tuple(k) for k in config.V2_SPLITS]:
        arms += ["BASE_noICULOS"] + list(V2_ARMS)
    return arms


def save_job(name, arm, key, horizon, model, metrics, test, extra):
    strategy, seed, fold = key
    rows = [{"arm": arm, "strategy": strategy, "seed": seed, "fold": fold, "horizon": horizon,
             "model": model, "metric": k, "value": v} for k, v in {**metrics, **extra}.items()]
    test[["patient_id", "ICULOS", "y", "p"]].to_parquet(os.path.join(config.PREDS_DIR, name + ".parquet"))
    pd.DataFrame(rows).to_csv(os.path.join(config.JOBS_DIR, name + ".csv"), index=False)


def collect_results():
    files = sorted(glob.glob(os.path.join(config.JOBS_DIR, "*.csv")))
    files = [f for f in files if not f.endswith("_importance.csv")]
    if not files:
        return pd.DataFrame()
    results = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    results.to_csv(config.RESULTS_CSV, index=False)
    return results


def run_arm(arm, matrix, data, roles, key):
    Xv, columns = matrix
    labels = data["labels"]
    train, val, test = roles == "train", roles == "val", roles == "test"
    has_val = val.any()
    todo = [(h, m) for h in config.HORIZONS for m in MODELS if not job_done(job_name(arm, key, h, m))]
    if not todo:
        return
    dtrain = xgb.DMatrix(Xv[train], feature_names=columns)
    dval = xgb.DMatrix(Xv[val], feature_names=columns) if has_val else None
    dtest = xgb.DMatrix(Xv[test], feature_names=columns)
    test_frame = labels.loc[test, ["patient_id", "ICULOS", "SepsisLabel", "onset_hour"]].copy()

    for horizon, model in todo:
        name = job_name(arm, key, horizon, model)
        y = labels[label_col(horizon)].to_numpy()
        start = time.time()
        extra = {}
        if model == "xgb":
            dtrain.set_label(y[train])
            if has_val:
                dval.set_label(y[val])
            booster, rounds = train_xgb(dtrain, dval, y[train])
            p_test = xgb_predict(booster, dtest, rounds)
            p_thr = xgb_predict(booster, dval if has_val else dtrain, rounds)
            extra["rounds"] = rounds
            if arm.startswith("NOVEL"):
                gain = pd.Series(booster.get_score(importance_type="gain"), name="gain")
                gain.to_csv(os.path.join(config.JOBS_DIR, name + "_importance.csv"))
        else:
            clf = train_logreg(Xv[train], y[train])
            p_test = clf.predict_proba(Xv[test])[:, 1]
            p_thr = clf.predict_proba(Xv[val] if has_val else Xv[train])[:, 1]
            extra["iterations"] = int(clf.n_iter_[0])
        # S2 has no validation set, so its threshold comes from the training predictions
        threshold = threshold_at_specificity(y[val] if has_val else y[train], p_thr)
        frame = test_frame.assign(y=y[test], p=p_test.astype("float32"))
        metrics = evaluate(frame, threshold, horizon)
        extra["train_seconds"] = time.time() - start
        extra["n_features"] = len(columns)
        save_job(name, arm, key, horizon, model, metrics, frame, extra)
        print(f"{name}: AUROC {metrics['auroc']:.4f} AUPRC {metrics['auprc']:.4f} "
              f"({extra['train_seconds']:.0f}s)")


def run(force=False, strategies=STRATEGIES):
    if force:
        for f in glob.glob(os.path.join(config.JOBS_DIR, "*")) + glob.glob(os.path.join(config.PREDS_DIR, "*")):
            os.remove(f)
    print("XGBoost device:", xgb_device())
    data = load_data()
    splits = n07.load_splits()
    for key in a01_make_splits.split_keys(splits):
        if key[0] not in strategies:
            continue
        arms = arms_for(key)
        names = [job_name(a, key, h, m) for a in arms for h in config.HORIZONS for m in MODELS]
        if all(job_done(n) for n in names):
            print("done:", key)
            continue
        print("split", key)
        roles = split_roles(data["labels"], key)
        assert not pd.isna(roles).any(), f"some rows have no role in {key}"
        base = base_matrix(data, roles, key)
        run_arm("BASE", base, data, roles, key)
        novel = novel_matrix(data, roles, key, base, config.TRANSDUCTIVE_P2)
        run_arm("NOVEL", novel, data, roles, key)
        del novel
        if ALT_P2_ARM in arms:
            novel = novel_matrix(data, roles, key, base, not config.TRANSDUCTIVE_P2)
            run_arm(ALT_P2_ARM, novel, data, roles, key)
            del novel
        if "BASE_noICULOS" in arms:
            run_arm("BASE_noICULOS", drop_base_columns(base, ["ICULOS"]), data, roles, key)
        for arm, opts in V2_ARMS.items():
            if arm not in arms:
                continue
            if all(job_done(job_name(arm, key, h, m)) for h in config.HORIZONS for m in MODELS):
                continue
            novel = novel_matrix(data, roles, key, base, config.TRANSDUCTIVE_P2, opts, arm)
            run_arm(arm, novel, data, roles, key)
            del novel
        collect_results()
    results = collect_results()
    print(f"{len(results)} result rows in {os.path.basename(config.RESULTS_CSV)}")
    return results


if __name__ == "__main__":
    run()
