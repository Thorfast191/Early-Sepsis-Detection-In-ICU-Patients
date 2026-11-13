import os

import matplotlib.pyplot as plt
import pandas as pd

from data_preprocessing import config, utils


def drop_nulls(df):
    missing = df.isna().mean()
    drop_cols = []
    for col in df.columns:
        if missing[col] > config.MISSING_THRESHOLD:
            drop_cols.append(col)
    df = df.drop(columns=drop_cols)

    vitals = utils.existing(config.VITALS, df)
    all_vitals_missing = df[vitals].isna().all(axis=1)
    df = df[~all_vitals_missing]
    return df


def run(force=False):
    before = utils.load(config.NODUP_PARQUET)

    if utils.is_done(config.DROPPED_PARQUET) and not force:
        print("loaded checkpoint:", os.path.basename(config.DROPPED_PARQUET))
        after = utils.load(config.DROPPED_PARQUET)
    else:
        after = drop_nulls(before)
        utils.save(after, config.DROPPED_PARQUET)

    missing_pct = (before.isna().mean() * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    colors = []
    for value in missing_pct.values:
        colors.append("tab:red" if value > config.MISSING_THRESHOLD * 100 else "tab:blue")

    plt.figure(figsize=(14, 5))
    plt.bar(missing_pct.index, missing_pct.values, color=colors)
    plt.axhline(config.MISSING_THRESHOLD * 100, color="red", linestyle="--",
                label=f"threshold {config.MISSING_THRESHOLD:.0%}")
    plt.xticks(rotation=90)
    plt.legend()
    utils.finish_plot("Missing values per column", "Column", "Missing (%)", "s03_missing_percent.png")

    dropped_cols = [c for c in before.columns if c not in after.columns]
    print(f"dropped {len(dropped_cols)} columns with more than {config.MISSING_THRESHOLD:.0%} missing:")
    print(", ".join(dropped_cols))

    vitals = utils.existing(config.VITALS, after)
    print(f"\ndropped {len(before) - len(after)} rows where all vitals {vitals} are missing")
    print("we do not drop every row with a null: almost every row is missing some lab value, "
          "so that would delete nearly the whole dataset")

    rows_with_null = after.isna().any(axis=1).mean() * 100
    print(f"rows that still have at least one null: {rows_with_null:.1f}% (handled by imputation in step 4)")
    print("\nshape before:", before.shape)
    print("shape after: ", after.shape)

    table = pd.DataFrame({"missing %": missing_pct.round(1)})
    table["dropped"] = [c in dropped_cols for c in table.index]
    utils.show(table)
    return after


if __name__ == "__main__":
    run()
