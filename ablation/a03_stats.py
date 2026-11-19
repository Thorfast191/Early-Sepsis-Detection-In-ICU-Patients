import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from ablation import a02_train_eval
from data_preprocessing import config, utils

CHUNK = 20
GROUP_COLS = ["strategy", "seed", "fold", "horizon", "model"]
METRICS = ["auroc", "auprc", "brier"]


def parse_name(path):
    arm, strategy, seed, fold, horizon, model = os.path.basename(path)[:-len(".parquet")].split("__")
    return {"arm": arm, "strategy": strategy, "seed": int(seed), "fold": int(fold),
            "horizon": int(horizon[1:]), "model": model, "path": path}


def prediction_index():
    files = glob.glob(os.path.join(config.PREDS_DIR, "*.parquet"))
    return pd.DataFrame([parse_name(f) for f in files])


def strategy_label(strategy, fold):
    if strategy != "S4":
        return strategy
    source, target = a02_train_eval.a01_make_splits.S4_DIRECTIONS[fold]
    return f"S4 {source}->{target}"


# ---- weighted metrics, one row of weights per bootstrap replicate ----

class SortedScores:
    """Predictions sorted once; metrics for many weight vectors via cumulative sums."""

    def __init__(self, y, p):
        order = np.argsort(p, kind="mergesort")
        self.order = order
        self.y = y[order].astype(bool)
        p_sorted = p[order]
        self.starts = np.r_[0, np.nonzero(np.diff(p_sorted))[0] + 1]
        self.p = p_sorted
        self.y_all = y
        self.p_all = p

    def metrics(self, w):
        """w: (B, n) row weights in the original row order. Returns dict of (B,) arrays."""
        w = w[:, self.order]
        wp = np.add.reduceat(np.where(self.y, w, 0.0), self.starts, axis=1)
        wn = np.add.reduceat(np.where(self.y, 0.0, w), self.starts, axis=1)
        total_p, total_n = wp.sum(1), wn.sum(1)
        # AUROC: each positive beats the negatives below it, ties count half
        below = np.cumsum(wn, axis=1) - wn
        with np.errstate(invalid="ignore", divide="ignore"):
            auroc = (wp * (below + 0.5 * wn)).sum(1) / (total_p * total_n)
            # average precision, thresholds from the highest score down
            tp = np.cumsum(wp[:, ::-1], axis=1)
            fp = np.cumsum(wn[:, ::-1], axis=1)
            precision = tp / (tp + fp)
            auprc = (wp[:, ::-1] * precision).sum(1) / total_p
        err = (self.p - self.y) ** 2
        brier = (w * err).sum(1) / w.sum(1)
        return {"auroc": auroc, "auprc": auprc, "brier": brier}


def patient_weights(codes, n_patients, rng, n_boot):
    """Bootstrap weights per row: how often its patient was drawn, for n_boot replicates."""
    for start in range(0, n_boot, CHUNK):
        b = min(CHUNK, n_boot - start)
        draws = rng.integers(0, n_patients, size=(b, n_patients))
        counts = np.stack([np.bincount(d, minlength=n_patients) for d in draws])
        yield counts[:, codes].astype("float64")


# ---- DeLong test for two correlated AUROCs (Sun and Xu 2014) ----

def delong(y, p1, p2):
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, v01, v10 = [], [], []
    for p in (p1, p2):
        tx, ty = rankdata(p[pos]), rankdata(p[neg])
        tz = rankdata(np.r_[p[pos], p[neg]])
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v01.append((tz[:m] - tx) / n)
        v10.append(1 - (tz[m:] - ty) / m)
    s = np.cov(np.array(v01)) / m + np.cov(np.array(v10)) / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var <= 0:
        return aucs[0] - aucs[1], np.nan
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs[0] - aucs[1], 2 * norm.sf(abs(z))


# ---- bootstrap over every test set ----

def bootstrap_all():
    """Bootstrap replicates of AUROC, AUPRC and Brier for every (arm, test set).

    BASE and NOVEL of the same test set get the same patient resamples, so the
    differences are paired.
    """
    index = prediction_index()
    replicates, points, delongs = {}, [], []
    for key, jobs in index.groupby(GROUP_COLS):
        frames = {row.arm: pd.read_parquet(row.path) for row in jobs.itertuples()}
        arms = sorted(frames)
        ref = frames[arms[0]]
        for arm in arms[1:]:
            same = frames[arm][["patient_id", "ICULOS", "y"]].reset_index(drop=True)
            assert same.equals(ref[["patient_id", "ICULOS", "y"]].reset_index(drop=True)), \
                f"{arm} and {arms[0]} have different test rows in {key}"
        y = ref["y"].to_numpy()
        codes, uniques = pd.factorize(ref["patient_id"])
        scores = {arm: SortedScores(y, frames[arm]["p"].to_numpy(dtype="float64")) for arm in arms}
        for arm in arms:
            point = scores[arm].metrics(np.ones((1, len(y))))
            points.append({**dict(zip(GROUP_COLS, key)), "arm": arm,
                           **{k: float(v[0]) for k, v in point.items()}})
        rng = np.random.default_rng(config.RANDOM_STATE)
        reps = {arm: {k: [] for k in METRICS} for arm in arms}
        for w in patient_weights(codes, len(uniques), rng, config.N_BOOTSTRAP):
            for arm in arms:
                for k, v in scores[arm].metrics(w).items():
                    reps[arm][k].append(v)
        for arm in arms:
            replicates[key + (arm,)] = {k: np.concatenate(v) for k, v in reps[arm].items()}
        if "BASE" in frames:
            for arm in arms:
                if arm != "BASE":
                    diff, p = delong(y, frames[arm]["p"].to_numpy(), frames["BASE"]["p"].to_numpy())
                    delongs.append({**dict(zip(GROUP_COLS, key)), "arm": arm, "delong_diff": diff,
                                    "delong_p": p})
    return replicates, pd.DataFrame(points), pd.DataFrame(delongs)


def ci(values):
    return np.nanpercentile(values, 2.5), np.nanpercentile(values, 97.5)


def boot_p(diffs):
    """Two-sided bootstrap p-value for a difference of zero."""
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return np.nan
    return min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))


def group_mean(replicates, points, strategy_rows):
    """Mean over the seeds/folds of a strategy: point estimate and bootstrap replicates."""
    keys = [tuple(r) for r in strategy_rows[GROUP_COLS + ["arm"]].itertuples(index=False)]
    pts = strategy_rows[METRICS].mean()
    reps = {k: np.nanmean(np.stack([replicates[key][k] for key in keys]), axis=0) for k in METRICS}
    return pts, reps


def strategy_groups(points):
    """Rows of the summary: S1, S2, S3, each S4 direction, and S4 as the mean of both."""
    points = points.copy()
    points["label"] = [strategy_label(s, f) for s, f in zip(points["strategy"], points["fold"])]
    groups = []
    for (label, arm, horizon, model), rows in points.groupby(["label", "arm", "horizon", "model"]):
        groups.append((label, arm, horizon, model, rows))
    for (arm, horizon, model), rows in points[points["strategy"] == "S4"].groupby(["arm", "horizon", "model"]):
        groups.append(("S4", arm, horizon, model, rows))
    return groups


OUTPUTS = [config.STATS_CI_CSV, config.STATS_PAIRED_CSV, config.STATS_GAP_CSV, config.SUMMARY_CSV]


def up_to_date():
    """True when every output exists and is newer than every prediction file."""
    if not all(utils.is_done(f) for f in OUTPUTS):
        return False
    preds = glob.glob(os.path.join(config.PREDS_DIR, "*.parquet"))
    newest = max((os.path.getmtime(f) for f in preds), default=0)
    return min(os.path.getmtime(f) for f in OUTPUTS) > newest


def show_outputs(summary, paired, gaps):
    print("summary (mean over splits [95% bootstrap CI], patients resampled):")
    utils.show(summary)
    print("\nNOVEL - BASE, paired on the same test patients:")
    utils.show(paired.round(4))
    print("\ngeneralisation gap (mean S1/S3 AUROC - S4 AUROC):")
    utils.show(gaps.round(4))


def run(force=False):
    if up_to_date() and not force:
        print("loaded checkpoints:", ", ".join(os.path.basename(f) for f in OUTPUTS))
        summary = pd.read_csv(config.SUMMARY_CSV, index_col=[0, 1, 2])
        paired = pd.read_csv(config.STATS_PAIRED_CSV)
        gaps = pd.read_csv(config.STATS_GAP_CSV)
        show_outputs(summary, paired, gaps)
        return summary, paired, gaps
    replicates, points, delongs = bootstrap_all()
    per_set = []
    for key, reps in replicates.items():
        row = dict(zip(GROUP_COLS + ["arm"], key))
        point = points[(points[GROUP_COLS + ["arm"]] == pd.Series(row)).all(axis=1)].iloc[0]
        for k in METRICS:
            lo, hi = ci(reps[k])
            row.update({k: point[k], f"{k}_lo": lo, f"{k}_hi": hi})
        per_set.append(row)
    per_set = pd.DataFrame(per_set)

    # mean over seeds/folds with a bootstrap CI of that mean
    means, mean_reps = [], {}
    for label, arm, horizon, model, rows in strategy_groups(points):
        pts, reps = group_mean(replicates, points, rows)
        mean_reps[(label, arm, horizon, model)] = reps
        row = {"strategy": label, "arm": arm, "horizon": horizon, "model": model, "n_test_sets": len(rows)}
        for k in METRICS:
            lo, hi = ci(reps[k])
            row.update({k: pts[k], f"{k}_lo": lo, f"{k}_hi": hi,
                        f"{k}_sd_across_splits": rows[k].std(ddof=1) if len(rows) > 1 else np.nan})
        means.append(row)
    means = pd.DataFrame(means)
    per_set["level"] = "test set"
    means["level"] = "mean over splits"
    pd.concat([means, per_set], ignore_index=True).to_csv(config.STATS_CI_CSV, index=False)

    # paired differences against BASE (and transductive vs inductive P2 on S4)
    paired = []
    for (label, arm, horizon, model), reps in mean_reps.items():
        pairs = [("BASE", arm)] if arm != "BASE" else []
        if arm == a02_train_eval.ALT_P2_ARM:
            pairs.append((arm, "NOVEL"))
        for ref_arm, new_arm in pairs:
            ref_reps = mean_reps.get((label, ref_arm, horizon, model))
            if ref_reps is None:
                continue
            row = {"strategy": label, "comparison": f"{new_arm} - {ref_arm}", "horizon": horizon, "model": model}
            ref_pt = means[(means["strategy"] == label) & (means["arm"] == ref_arm)
                           & (means["horizon"] == horizon) & (means["model"] == model)].iloc[0]
            new_pt = means[(means["strategy"] == label) & (means["arm"] == new_arm)
                           & (means["horizon"] == horizon) & (means["model"] == model)].iloc[0]
            for k in ["auroc", "auprc"]:
                d = mean_reps[(label, new_arm, horizon, model)][k] - ref_reps[k]
                lo, hi = ci(d)
                row.update({f"{k}_diff": new_pt[k] - ref_pt[k], f"{k}_diff_lo": lo, f"{k}_diff_hi": hi,
                            f"{k}_boot_p": boot_p(d)})
            if ref_arm == "BASE" and len(delongs):
                d = delongs[(delongs["arm"] == new_arm) & (delongs["horizon"] == horizon)
                            & (delongs["model"] == model)]
                d = d[[strategy_label(s, f) == label or (label == "S4" and s == "S4")
                       for s, f in zip(d["strategy"], d["fold"])]]
                row["delong_p_per_split"] = ";".join(f"{p:.3g}" for p in d["delong_p"])
                row["delong_splits_p_below_0.05"] = int((d["delong_p"] < 0.05).sum())
                row["n_splits"] = len(d)
            paired.append(row)
    paired = pd.DataFrame(paired)
    paired.to_csv(config.STATS_PAIRED_CSV, index=False)

    # generalisation gap: mean of the S1 and S3 AUROC minus the S4 AUROC
    gaps = []
    for (arm, horizon, model), _ in means.groupby(["arm", "horizon", "model"]):
        need = [(s, arm, horizon, model) for s in ["S1", "S3", "S4"]]
        if not all(k in mean_reps for k in need):
            continue
        internal = (mean_reps[need[0]]["auroc"] + mean_reps[need[1]]["auroc"]) / 2
        gap_reps = internal - mean_reps[need[2]]["auroc"]
        pt = {s: means[(means["strategy"] == s) & (means["arm"] == arm) & (means["horizon"] == horizon)
                       & (means["model"] == model)]["auroc"].iloc[0] for s in ["S1", "S3", "S4"]}
        internal_pt = (pt["S1"] + pt["S3"]) / 2
        lo, hi = ci(gap_reps)
        gaps.append({"arm": arm, "horizon": horizon, "model": model, "internal_auroc": internal_pt,
                     "external_auroc": pt["S4"], "gap": internal_pt - pt["S4"], "gap_lo": lo, "gap_hi": hi})
    gaps = pd.DataFrame(gaps)
    gaps.to_csv(config.STATS_GAP_CSV, index=False)

    summary = summary_table(means)
    summary.to_csv(config.SUMMARY_CSV)
    show_outputs(summary, paired, gaps)
    return summary, paired, gaps


def summary_table(means):
    """rows = model x strategy x arm, columns = metric per horizon as mean [CI]."""
    results = a02_train_eval.collect_results()
    results["label"] = [strategy_label(s, f) for s, f in zip(results["strategy"], results["fold"])]
    extra = {}
    for metric in ["sensitivity", "ppv", "utility"]:
        sub = results[results["metric"] == metric]
        for (label, arm, horizon, model), v in sub.groupby(["label", "arm", "horizon", "model"])["value"]:
            extra[(label, arm, horizon, model, metric)] = v
        for (arm, horizon, model), v in sub[sub["strategy"] == "S4"].groupby(["arm", "horizon", "model"])["value"]:
            extra[("S4", arm, horizon, model, metric)] = v

    order = ["S1", "S2", "S3", "S4", "S4 A->B", "S4 B->A"]
    rows = {}
    for r in means.itertuples():
        idx = (r.model, r.strategy, r.arm)
        cells = rows.setdefault(idx, {})
        for k in ["auroc", "auprc"]:
            cells[f"{k.upper()} H{r.horizon}"] = (f"{getattr(r, k):.3f} "
                                                 f"[{getattr(r, k + '_lo'):.3f}, {getattr(r, k + '_hi'):.3f}]")
        for metric in ["sensitivity", "ppv", "utility"]:
            v = extra.get((r.strategy, r.arm, r.horizon, r.model, metric))
            if v is not None and v.notna().any():
                sd = f" ± {v.std(ddof=1):.3f}" if len(v) > 1 else ""
                cells[f"{metric} H{r.horizon}"] = f"{v.mean():.3f}{sd}"
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index = pd.MultiIndex.from_tuples(table.index, names=["model", "strategy", "arm"])
    columns = [f"{m} H{h}" for m in ["AUROC", "AUPRC", "sensitivity", "ppv", "utility"]
               for h in config.HORIZONS if f"{m} H{h}" in table.columns]
    table = table[columns]
    key = [(m, order.index(s) if s in order else 99, a) for m, s, a in table.index]
    return table.iloc[sorted(range(len(key)), key=lambda i: key[i])]


if __name__ == "__main__":
    run()
