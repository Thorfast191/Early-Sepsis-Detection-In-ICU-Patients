import os

import matplotlib.pyplot as plt
import pandas as pd

from data_preprocessing import config, utils

TOP_N = 4


def iqr_table(df, columns):
    rows = []
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        rows.append({
            "feature": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower": lower,
            "upper": upper,
            "outliers": n_outliers,
            "outlier %": n_outliers / len(df) * 100,
        })
    table = pd.DataFrame(rows).set_index("feature")
    return table.sort_values("outlier %", ascending=False)


def before():
    df = utils.load(config.CLEAN_PARQUET)
    utils.heading("Before: the IQR table uses all rows (no sample)")
    utils.preview_sample(df, utils.numeric_feature_cols(df))


def run(force=False):
    df = utils.load(config.CLEAN_PARQUET)
    columns = utils.numeric_feature_cols(df)

    table = iqr_table(df, columns)
    utils.show(table.round(2))
    csv_path = os.path.join(config.FIGURES_DIR, "s09_iqr_outliers.csv")
    table.to_csv(csv_path)
    print("saved", csv_path)

    top = list(table.index[:TOP_N])
    fig, axes = plt.subplots(1, len(top), figsize=(4 * len(top), 5))
    for ax, col in zip(axes, top):
        ax.boxplot(df[col], flierprops={"markersize": 2, "alpha": 0.3})
        ax.axhline(table.loc[col, "lower"], color="red", linestyle="--", label="IQR bounds")
        ax.axhline(table.loc[col, "upper"], color="red", linestyle="--")
        ax.set_title(f"{col} ({table.loc[col, 'outlier %']:.1f}% outliers)")
        ax.set_xlabel(col)
        ax.set_ylabel("Value")
        ax.set_xticks([])
        ax.legend(loc="best")
    utils.finish_plot(f"Top {len(top)} features by IQR outlier %", None, None, "s09_iqr_top.png")

    print("outliers are NOT removed: extreme vitals and labs (very high HR, high lactate, "
          "fever) are often exactly what sepsis looks like, so they stay in the data")


if __name__ == "__main__":
    run()
