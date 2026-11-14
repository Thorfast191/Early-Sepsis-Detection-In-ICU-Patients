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


def in_notebook():
    """Return True when running inside Jupyter or Colab."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def show(df):
    """Show a table nicely in a notebook, or print it in a terminal."""
    if in_notebook():
        from IPython.display import display
        display(df)
    else:
        print(df.to_string())


def heading(text):
    """Print a plain heading such as 'Before:' above a table."""
    print(f"\n{text}")


def show_highlighted(df, mask, color):
    """Show df with the cells where mask is True coloured (notebook only).

    Outside a notebook the table is printed without colours.
    """
    if not in_notebook():
        print(df.round(2).to_string())
        return
    from IPython.display import display

    def colour_cells(table):
        styles = pd.DataFrame("", index=table.index, columns=table.columns)
        styles[mask.values] = f"background-color: {color}"
        return styles

    display(df.style.apply(colour_cells, axis=None).format(precision=2))


# chosen once per session so every step shows the same patient
DEMO_PATIENT = None


def is_demo_candidate(rows):
    """True if one patient's rows are a good demo: sepsis, 12+ hours, Temp missing early."""
    rows = rows.sort_values("ICULOS")
    if rows["SepsisLabel"].max() != 1 or len(rows) < 12:
        return False
    return bool(rows["Temp"].head(6).isna().any())


def pick_demo_patient(df):
    """Return the demo patient id, choosing it from df the first time.

    The demo patient is the first patient id (sorted) that passes
    is_demo_candidate. Returns None if df has no such patient.
    """
    global DEMO_PATIENT
    if DEMO_PATIENT is not None:
        return DEMO_PATIENT
    sepsis_ids = sorted(df.loc[df["SepsisLabel"] == 1, "patient_id"].unique())
    for patient_id in sepsis_ids:
        if is_demo_candidate(df[df["patient_id"] == patient_id]):
            DEMO_PATIENT = patient_id
            print("demo patient:", patient_id)
            return patient_id
    return None


def demo_patient():
    """The demo patient, picked from sepsis_raw.parquet if not chosen yet.

    Later files are already imputed (no missing Temp), so the choice is
    always made on the raw data to keep the same patient in every step.
    """
    if DEMO_PATIENT is not None:
        return DEMO_PATIENT
    raw = pd.read_parquet(config.RAW_PARQUET, columns=["patient_id", "ICULOS", "SepsisLabel", "Temp"])
    return pick_demo_patient(raw)


def demo_rows(df, patient_id, n=10, columns=None, hours=None):
    """One patient's rows sorted by ICULOS: the first n hours, or the given ICULOS hours.

    Rows are matched by patient_id and ICULOS, not by index, because step 4
    sorts the data and resets the index. ICULOS is always the first column.
    """
    rows = df[df["patient_id"] == patient_id].sort_values("ICULOS")
    if hours is not None:
        rows = rows[rows["ICULOS"].isin(hours)]
    else:
        rows = rows.head(n)
    if columns is not None:
        rows = rows[["ICULOS"] + [c for c in columns if c != "ICULOS"]]
    return rows


def preview_sample(df, columns, n_per_class=None):
    """Show what a plot step uses: rows per class and 3 rows of each class.

    With n_per_class the balanced sample is shown, otherwise all rows.
    """
    if n_per_class is not None:
        df = balanced_sample(df, n_per_class)
    counts = df["SepsisLabel"].value_counts().sort_index()
    print("rows per class:", {int(k): int(v) for k, v in counts.items()})
    for label in counts.index:
        heading(f"SepsisLabel = {label}, first 3 rows:")
        show(df[df["SepsisLabel"] == label][columns].head(3))


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
