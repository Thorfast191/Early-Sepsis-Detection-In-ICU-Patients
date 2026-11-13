import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import config, utils

COLUMNS = ["HR", "Temp", "Resp", "MAP"]


def run(force=False):
    df = utils.load(config.CLEAN_PARQUET)
    sample = utils.balanced_sample(df, config.PLOT_SAMPLE_PER_CLASS)
    columns = utils.existing(COLUMNS, sample)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, col in zip(axes.flat, columns):
        sns.violinplot(data=sample, x="SepsisLabel", y=col, hue="SepsisLabel",
                       palette={0: "tab:blue", 1: "tab:red"}, legend=False, ax=ax)
        ax.set_title(f"{col} by SepsisLabel")
        ax.set_xlabel("SepsisLabel")
        ax.set_ylabel(col)
    utils.finish_plot("Violin plots of vitals", None, None, "s11_violin.png")


if __name__ == "__main__":
    run()
