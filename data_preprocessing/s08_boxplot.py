import math

import matplotlib.pyplot as plt

from data_preprocessing import config, utils


def before():
    df = utils.load(config.CLEAN_PARQUET)
    utils.heading("Before: the boxplots use all rows (no sample)")
    utils.preview_sample(df, utils.numeric_feature_cols(df))


def run(force=False):
    df = utils.load(config.CLEAN_PARQUET)
    columns = utils.numeric_feature_cols(df)
    print(f"boxplots for {len(columns)} numeric features on all {len(df)} rows")

    n_cols = 5
    n_rows = math.ceil(len(columns) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.2 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        ax.boxplot(df[col], flierprops={"markersize": 2, "alpha": 0.3})
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Value")
        ax.set_xticks([])

    # hide empty grid cells
    for j in range(len(columns), len(axes)):
        fig.delaxes(axes[j])

    utils.finish_plot("Boxplots of all numeric features", None, None, "s08_boxplots.png")


if __name__ == "__main__":
    run()
