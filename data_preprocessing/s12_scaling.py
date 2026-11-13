import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_preprocessing import config, utils


def scale(df, columns):
    # NOTE: this scaler is fit on all rows only to show the step. In the
    # modelling part the scaler must be refit on the training hospital only,
    # otherwise test statistics leak into training.
    scaler = StandardScaler()
    scaled = df.copy()
    scaled[columns] = scaler.fit_transform(df[columns]).astype("float32")
    return scaled


def run(force=False):
    before = utils.load(config.CLEAN_PARQUET)
    columns = utils.numeric_feature_cols(before)

    if utils.is_done(config.SCALED_PARQUET) and not force:
        print("loaded checkpoint:", os.path.basename(config.SCALED_PARQUET))
        after = utils.load(config.SCALED_PARQUET)
    else:
        after = scale(before, columns)
        utils.save(after, config.SCALED_PARQUET)

    print(f"StandardScaler applied to {len(columns)} columns:")
    print(", ".join(columns))
    print("\nbefore scaling:")
    utils.show(before[columns].describe().T[["mean", "std", "min", "max"]].round(2))
    print("\nafter scaling (mean ~0, std ~1):")
    utils.show(after[columns].describe().T[["mean", "std", "min", "max"]].round(2))

    if "HR" not in columns:
        return after

    # MinMaxScaler only for comparison, it is not saved
    hr_minmax = MinMaxScaler().fit_transform(before[["HR"]]).ravel()
    plot_df = pd.DataFrame({
        "original": before["HR"].values,
        "standard": after["HR"].values,
        "minmax": hr_minmax,
    }).sample(n=min(20000, len(before)), random_state=config.RANDOM_STATE)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    sns.kdeplot(x=plot_df["original"], fill=True, ax=axes[0])
    axes[0].set_title("HR before scaling")
    axes[0].set_xlabel("HR (beats per minute)")
    sns.kdeplot(x=plot_df["standard"], fill=True, color="tab:green", ax=axes[1])
    axes[1].set_title("HR after StandardScaler")
    axes[1].set_xlabel("HR (z-score)")
    sns.kdeplot(x=plot_df["minmax"], fill=True, color="tab:orange", ax=axes[2])
    axes[2].set_title("HR after MinMaxScaler (comparison)")
    axes[2].set_xlabel("HR (0 to 1)")
    for ax in axes:
        ax.set_ylabel("Density")
    utils.finish_plot("Effect of scaling on HR", None, None, "s12_scaling_hr.png")
    return after


if __name__ == "__main__":
    run()
