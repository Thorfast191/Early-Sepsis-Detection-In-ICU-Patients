import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import config, utils

PAIRS = [("HR", "Resp"), ("MAP", "HR")]


def run(force=False):
    df = utils.load(config.CLEAN_PARQUET)
    sample = utils.balanced_sample(df, config.PLOT_SAMPLE_PER_CLASS)
    print(f"plotting a balanced sample of {len(sample)} rows "
          f"({config.PLOT_SAMPLE_PER_CLASS} per class at most)")

    for x, y in PAIRS:
        if x not in sample.columns or y not in sample.columns:
            print(f"skipping {x} vs {y}, column was dropped")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.scatterplot(data=sample, x=x, y=y, alpha=0.2, s=10, ax=axes[0])
        axes[0].set_title(f"{x} vs {y}")
        sns.scatterplot(data=sample, x=x, y=y, hue="SepsisLabel", alpha=0.2, s=10,
                        palette={0: "tab:blue", 1: "tab:red"}, ax=axes[1])
        axes[1].set_title(f"{x} vs {y} by SepsisLabel")
        for ax in axes:
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        utils.finish_plot(f"Scatter plot: {x} vs {y}", None, None, f"s05_scatter_{x}_{y}.png")


if __name__ == "__main__":
    run()
