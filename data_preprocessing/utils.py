import os

import matplotlib.pyplot as plt
import pandas as pd

from data_preprocessing import config


def load(path):
    """Read a parquet file into a DataFrame."""
    return pd.read_parquet(path)


def save(df, path):
    """Save a DataFrame as parquet and print its shape."""
    df.to_parquet(path, index=True)
    print(f"saved {os.path.basename(path)} with shape {df.shape}")


def is_done(path):
    """Return True if the checkpoint file already exists."""
    return os.path.exists(path)


def show(df):
    """Show a table nicely in a notebook, or print it in a terminal."""
    try:
        from IPython.display import display
        from IPython import get_ipython
        if get_ipython() is not None:
            display(df)
            return
    except ImportError:
        pass
    print(df.to_string())


def balanced_sample(df, n_per_class):
    """Take up to n_per_class rows from each SepsisLabel class."""
    parts = []
    for label, group in df.groupby("SepsisLabel"):
        n = min(n_per_class, len(group))
        parts.append(group.sample(n=n, random_state=config.RANDOM_STATE))
    return pd.concat(parts)


def finish_plot(title, xlabel, ylabel, filename):
    """Add title and axis labels, save the figure as png in figures/ and show it.

    With one axis the labels go on that axis. With several subplots the title
    becomes the main title and each subplot must already have its own labels.
    """
    fig = plt.gcf()
    if len(fig.axes) == 1:
        ax = fig.axes[0]
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_DIR, filename), dpi=100, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def numeric_feature_cols(df):
    """Numeric feature columns, without the label, ids and binary/one-hot columns."""
    skip = ["SepsisLabel", "patient_id", "hospital"]
    cols = []
    for col in df.columns:
        if col in skip:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].nunique() <= 2:
            continue
        cols.append(col)
    return cols


def existing(cols, df):
    """Keep only the columns that are still in df."""
    return [c for c in cols if c in df.columns]
