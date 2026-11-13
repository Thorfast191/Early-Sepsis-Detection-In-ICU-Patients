import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import config, utils

COLUMNS = ["HR", "Temp", "Resp", "Age"]


def run(force=False):
    df = utils.load(config.CLEAN_PARQUET)
    sample = utils.balanced_sample(df, config.PLOT_SAMPLE_PER_CLASS)
    columns = utils.existing(COLUMNS, sample)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flat, columns):
        sns.kdeplot(data=sample, x=col, fill=True, ax=ax)
        ax.set_title(f"KDE of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
    utils.finish_plot("KDE plots (all rows)", None, None, "s06_kde.png")

    # common_norm=False scales each class on its own, otherwise the
    # small sepsis class would look like a flat line
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flat, columns):
        sns.kdeplot(data=sample, x=col, hue="SepsisLabel", common_norm=False, fill=True,
                    palette={0: "tab:blue", 1: "tab:red"}, ax=ax)
        ax.set_title(f"KDE of {col} by SepsisLabel")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
    utils.finish_plot("KDE plots by SepsisLabel", None, None, "s06_kde_by_label.png")


if __name__ == "__main__":
    run()
