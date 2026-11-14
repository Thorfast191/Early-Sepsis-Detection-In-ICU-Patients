import warnings

import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import config, utils

COLUMNS = ["HR", "Temp", "Resp", "MAP"]
PALETTE = {0: "tab:blue", 1: "tab:red"}


def before():
    df = utils.load(config.CLEAN_PARQUET)
    columns = utils.existing(COLUMNS + ["SepsisLabel"], df)
    utils.heading("Before: small balanced sample used for the swarm plots")
    utils.preview_sample(df, columns, config.SWARM_SAMPLE_PER_CLASS)
    utils.heading("Before: balanced sample used for the strip plots")
    utils.preview_sample(df, columns, config.PLOT_SAMPLE_PER_CLASS)


def run(force=False):
    df = utils.load(config.CLEAN_PARQUET)
    columns = utils.existing(COLUMNS, df)

    # swarm plots get very slow with many points, so use a small sample
    small = utils.balanced_sample(df, config.SWARM_SAMPLE_PER_CLASS)
    print(f"swarm plot on {len(small)} rows ({config.SWARM_SAMPLE_PER_CLASS} per class)")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    with warnings.catch_warnings():
        # seaborn warns when a few points can't be placed; that is fine here
        warnings.simplefilter("ignore", UserWarning)
        for ax, col in zip(axes.flat, columns):
            sns.swarmplot(data=small, x="SepsisLabel", y=col, hue="SepsisLabel",
                          palette=PALETTE, size=2.5, legend=False, ax=ax)
            ax.set_title(f"{col} by SepsisLabel")
            ax.set_xlabel("SepsisLabel")
            ax.set_ylabel(col)
    utils.finish_plot("Swarm plots of vitals", None, None, "s10_swarm.png")

    sample = utils.balanced_sample(df, config.PLOT_SAMPLE_PER_CLASS)
    print(f"strip plot on {len(sample)} rows ({config.PLOT_SAMPLE_PER_CLASS} per class)")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, col in zip(axes.flat, columns):
        sns.stripplot(data=sample, x="SepsisLabel", y=col, hue="SepsisLabel",
                      palette=PALETTE, size=2, alpha=0.2, jitter=0.3, legend=False, ax=ax)
        ax.set_title(f"{col} by SepsisLabel")
        ax.set_xlabel("SepsisLabel")
        ax.set_ylabel(col)
    utils.finish_plot("Strip plots of vitals", None, None, "s10_strip.png")


if __name__ == "__main__":
    run()
