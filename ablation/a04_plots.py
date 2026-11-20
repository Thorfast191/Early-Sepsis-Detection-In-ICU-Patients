import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from ablation import a02_train_eval, a03_stats
from data_preprocessing import config, utils

STRATEGY_ORDER = ["S1", "S2", "S3", "S4"]
COLORS = {"BASE": "tab:blue", "NOVEL": "tab:red", a02_train_eval.ALT_P2_ARM: "tab:orange"}
MAIN_MODEL = "xgb"


def fig_name(name):
    return config.NOVEL_FIG_PREFIX + name


def load_means():
    stats = pd.read_csv(config.STATS_CI_CSV)
    return stats[stats["level"] == "mean over splits"]


def bars_by_strategy(means, horizon):
    sub = means[(means["horizon"] == horizon) & (means["model"] == MAIN_MODEL)]
    arms = [a for a in COLORS if a in set(sub["arm"])]
    x = np.arange(len(STRATEGY_ORDER))
    width = 0.8 / len(arms)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ["auroc", "auprc"]):
        for i, arm in enumerate(arms):
            rows = sub[sub["arm"] == arm].set_index("strategy").reindex(STRATEGY_ORDER)
            err = np.vstack([rows[metric] - rows[f"{metric}_lo"], rows[f"{metric}_hi"] - rows[metric]])
            ax.bar(x + (i - (len(arms) - 1) / 2) * width, rows[metric], width, yerr=err, capsize=3,
                   label=arm, color=COLORS[arm])
        ax.set_xticks(x, STRATEGY_ORDER)
        ax.set_title(f"{metric.upper()}, horizon {horizon} h")
        ax.set_xlabel("Splitting strategy")
        ax.set_ylabel(f"{metric.upper()} (95% bootstrap CI)")
        ax.legend()
    utils.finish_plot(f"BASE vs NOVEL by splitting strategy (XGBoost, {horizon} h horizon)", None, None,
                      fig_name(f"a04_bars_H{horizon}.png"))


def gap_bars():
    gaps = pd.read_csv(config.STATS_GAP_CSV)
    gaps = gaps[(gaps["model"] == MAIN_MODEL) & (gaps["internal"] == "mean(S1,S3)")]
    arms = [a for a in COLORS if a in set(gaps["arm"])]
    x = np.arange(len(config.HORIZONS))
    width = 0.8 / len(arms)
    plt.figure(figsize=(10, 5))
    for i, arm in enumerate(arms):
        rows = gaps[gaps["arm"] == arm].set_index("horizon").reindex(config.HORIZONS)
        err = np.vstack([rows["gap"] - rows["gap_lo"], rows["gap_hi"] - rows["gap"]])
        plt.bar(x + (i - (len(arms) - 1) / 2) * width, rows["gap"], width, yerr=err, capsize=3,
                label=arm, color=COLORS[arm])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, [f"{h} h" for h in config.HORIZONS])
    plt.legend()
    utils.finish_plot("Generalisation gap: mean(S1, S3) AUROC - S4 AUROC (XGBoost)",
                      "Prediction horizon", "AUROC gap (95% bootstrap CI)", fig_name("a04_generalisation_gap.png"))


def pred_path(arm, strategy, seed, fold, horizon, model=MAIN_MODEL):
    name = a02_train_eval.job_name(arm, (strategy, seed, fold), horizon, model)
    return os.path.join(config.PREDS_DIR, name + ".parquet")


def curves(horizon):
    seed = config.SPLIT_SEEDS[0]
    panels = [("S1 test", "S1", 0), ("S4 A->B", "S4", 0), ("S4 B->A", "S4", 1)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for col, (title, strategy, fold) in enumerate(panels):
        for arm in ["BASE", "NOVEL"]:
            path = pred_path(arm, strategy, seed, fold, horizon)
            if not os.path.exists(path):
                continue
            pred = pd.read_parquet(path)
            fpr, tpr, _ = roc_curve(pred["y"], pred["p"])
            precision, recall, _ = precision_recall_curve(pred["y"], pred["p"])
            axes[0, col].plot(fpr, tpr, label=arm, color=COLORS[arm])
            axes[1, col].plot(recall, precision, label=arm, color=COLORS[arm])
        axes[0, col].plot([0, 1], [0, 1], "k--", linewidth=0.8)
        axes[0, col].set_title(f"ROC, {title}")
        axes[0, col].set_xlabel("False positive rate")
        axes[0, col].set_ylabel("True positive rate")
        axes[1, col].set_title(f"Precision-recall, {title}")
        axes[1, col].set_xlabel("Recall")
        axes[1, col].set_ylabel("Precision")
        for ax in axes[:, col]:
            ax.legend()
    utils.finish_plot(f"ROC and PR curves, BASE vs NOVEL (XGBoost, {horizon} h, seed {seed})", None, None,
                      fig_name(f"a04_curves_H{horizon}.png"))


def auroc_vs_horizon(means):
    sub = means[means["model"] == MAIN_MODEL]
    fig, axes = plt.subplots(1, len(STRATEGY_ORDER), figsize=(18, 4.5), sharey=True)
    for ax, strategy in zip(axes, STRATEGY_ORDER):
        for arm in [a for a in COLORS if a in set(sub["arm"])]:
            rows = sub[(sub["strategy"] == strategy) & (sub["arm"] == arm)].sort_values("horizon")
            if rows.empty:
                continue
            ax.errorbar(rows["horizon"], rows["auroc"],
                        yerr=[rows["auroc"] - rows["auroc_lo"], rows["auroc_hi"] - rows["auroc"]],
                        marker="o", capsize=3, label=arm, color=COLORS[arm])
        ax.set_xticks(config.HORIZONS)
        ax.set_title(strategy)
        ax.set_xlabel("Prediction horizon (hours)")
        ax.set_ylabel("AUROC")
        ax.legend()
    utils.finish_plot("AUROC vs prediction horizon (XGBoost)", None, None, fig_name("a04_auroc_vs_horizon.png"))


def importances(top=20):
    files = glob.glob(os.path.join(config.JOBS_DIR, "NOVEL__*__xgb_importance.csv"))
    if not files:
        print("no NOVEL XGBoost importances yet")
        return None
    shares = []
    for f in files:
        gain = pd.read_csv(f, index_col=0)["gain"]
        shares.append(gain / gain.sum())
    # mean share of total gain over all NOVEL XGBoost models (absent feature = 0)
    mean = pd.concat(shares, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False)
    mean.to_frame("mean_gain_share").to_csv(config.IMPORTANCE_CSV)
    head = mean.head(top)[::-1]
    plt.figure(figsize=(9, 7))
    plt.barh(head.index, head.values, color="tab:red")
    utils.finish_plot(f"Top {top} XGBoost gain importances, NOVEL arm ({len(files)} models)",
                      "Mean share of total gain", "Feature", fig_name("a04_importance_novel.png"))
    return mean


def v2_arm_bars(means):
    """External (S4) and internal (S1) AUROC of every arm, including the v2 arms."""
    sub = means[(means["model"] == MAIN_MODEL)]
    v2 = [a for a in sub["arm"].unique() if a not in COLORS]
    if not v2:
        return
    fig, axes = plt.subplots(1, len(config.HORIZONS), figsize=(22, 8), sharey=True)
    for ax, horizon in zip(axes, config.HORIZONS):
        ext = sub[(sub["strategy"] == "S4") & (sub["horizon"] == horizon)].set_index("arm")
        order = ext["auroc"].sort_values().index
        ext = ext.loc[order]
        err = np.vstack([ext["auroc"] - ext["auroc_lo"], ext["auroc_hi"] - ext["auroc"]])
        colors = [COLORS.get(a, "tab:green" if a.startswith("NOVEL_fix") else "tab:gray") for a in order]
        ax.barh(order, ext["auroc"], xerr=err, color=colors, capsize=2)
        base = ext.loc["BASE", "auroc"] if "BASE" in ext.index else None
        if base is not None:
            ax.axvline(base, color="tab:blue", linestyle="--", linewidth=1)
        ax.set_xlim(0.4, max(0.8, float(ext["auroc_hi"].max()) + 0.02))
        ax.set_title(f"S4 (A->B and B->A), {horizon} h")
        ax.set_xlabel("AUROC (95% bootstrap CI)")
        ax.set_ylabel("Arm")
    utils.finish_plot("External AUROC of every arm (XGBoost); dashed line = BASE", None, None,
                      fig_name("a04_v2_arms_external.png"))


def screening_plot():
    if not utils.is_done(config.SCREEN_CSV):
        return
    table = pd.read_csv(config.SCREEN_CSV)
    table = table[(table["rule"] == "domain_auroc") & (table["arm"] == "NOVEL_fix_screen")]
    if table.empty:
        return
    keys = table[["strategy", "fold"]].drop_duplicates().itertuples(index=False)
    keys = list(keys)
    fig, axes = plt.subplots(1, len(keys), figsize=(7 * len(keys), 12))
    axes = np.atleast_1d(axes)
    for ax, (strategy, fold) in zip(axes, keys):
        t = table[(table["strategy"] == strategy) & (table["fold"] == fold)].sort_values("score")
        colors = ["tab:red" if d else "tab:green" for d in t["dropped"]]
        ax.barh(t["unit"], t["score"], color=colors)
        ax.axvline(config.SCREEN_MAX_DOMAIN_AUROC, color="black", linestyle="--")
        ax.set_xlim(0.4, 1.0)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(f"{strategy} fold {fold}: red = dropped")
        ax.set_xlabel("Hospital-classifier AUROC (no sepsis labels)")
        ax.set_ylabel("Feature unit")
    utils.finish_plot("Site-invariance screening: how well each feature unit identifies the hospital",
                      None, None, fig_name("a04_v2_screening.png"))


def run(force=False):
    means = load_means()
    for horizon in config.HORIZONS:
        bars_by_strategy(means, horizon)
    gap_bars()
    for horizon in config.HORIZONS:
        curves(horizon)
    auroc_vs_horizon(means)
    v2_arm_bars(means)
    screening_plot()
    top = importances()
    if top is not None:
        utils.show(top.head(20).to_frame("mean gain share").round(4))


if __name__ == "__main__":
    run()
