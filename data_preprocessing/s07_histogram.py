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
        sns.histplot(data=sample, x=col, bins=40, ax=ax)
        ax.set_title(f"Histogram of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
    utils.finish_plot("Histograms (all rows)", None, None, "s07_histogram.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flat, columns):
        sns.histplot(data=sample, x=col, hue="SepsisLabel", bins=40, stat="density",
                     common_norm=False, palette={0: "tab:blue", 1: "tab:red"}, ax=ax)
        ax.set_title(f"Histogram of {col} by SepsisLabel")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
    utils.finish_plot("Histograms by SepsisLabel", None, None, "s07_histogram_by_label.png")


if __name__ == "__main__":
    run()
