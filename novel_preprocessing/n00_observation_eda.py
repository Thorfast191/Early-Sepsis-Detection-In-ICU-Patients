import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from data_preprocessing import config, utils
from novel_preprocessing import common


def measurement_rates(grid, septic):
    """Measurements per patient-hour for each variable, by hospital and by sepsis status."""
    observed = grid[config.MEASURED].notna()
    groups = {
        "hospital A": grid["hospital"] == "A",
        "hospital B": grid["hospital"] == "B",
        "septic": grid["patient_id"].map(septic) == 1,
        "non-septic": grid["patient_id"].map(septic) == 0,
    }
    table = pd.DataFrame({name: observed[mask].mean() for name, mask in groups.items()})
    table["A / B"] = table["hospital A"] / table["hospital B"]
    table["kind"] = ["vital" if v in config.VITALS_ALL else "lab" for v in table.index]
    return table


def measurement_gaps(grid):
    """Hours between consecutive measurements of the same patient, per variable and hospital."""
    rows = []
    for v in config.MEASURED:
        obs = grid.loc[grid[v].notna(), ["patient_id", "hospital", "ICULOS"]]
        gap = obs.groupby("patient_id", sort=False)["ICULOS"].diff()
        for hospital in ["A", "B"]:
            g = gap[(obs["hospital"] == hospital).to_numpy()].dropna()
            rows.append({"variable": v, "hospital": hospital, "n_gaps": len(g),
                         "median_h": g.median() if len(g) else np.nan,
                         "mean_h": g.mean() if len(g) else np.nan,
                         "p90_h": g.quantile(0.9) if len(g) else np.nan})
    return pd.DataFrame(rows)


def ks_vitals(grid):
    rows = []
    for v in config.VITALS_ALL:
        a = grid.loc[grid["hospital"] == "A", v].dropna()
        b = grid.loc[grid["hospital"] == "B", v].dropna()
        if len(a) == 0 or len(b) == 0:
            rows.append({"vital": v, "n_A": len(a), "n_B": len(b), "KS": np.nan, "p_value": np.nan})
            continue
        result = ks_2samp(a, b)
        rows.append({"vital": v, "n_A": len(a), "n_B": len(b), "median_A": a.median(),
                     "median_B": b.median(), "KS": result.statistic, "p_value": result.pvalue})
    return pd.DataFrame(rows).set_index("vital")


def grouped_bars(table, columns, title, ylabel, filename, log=False):
    variables = list(table.index)
    x = np.arange(len(variables))
    width = 0.8 / len(columns)
    plt.figure(figsize=(16, 5))
    for i, col in enumerate(columns):
        plt.bar(x + (i - (len(columns) - 1) / 2) * width, table[col], width, label=col)
    plt.xticks(x, variables, rotation=90)
    if log:
        plt.yscale("log")
    plt.legend()
    utils.finish_plot(title, "Variable", ylabel, config.NOVEL_FIG_PREFIX + filename)


def run(force=False):
    table = common.patient_table()
    patients = common.selected_patients(table)
    grid = common.load_grid(config.MEASURED, patients)
    grid = grid[grid["s03_kept"]]
    septic = table["ever_septic"]

    rates = measurement_rates(grid, septic)
    rates.to_csv(config.EDA_RATES_CSV)
    print("measurements per patient-hour:")
    utils.show(rates.round(4))
    labs = rates[rates["kind"] == "lab"]
    print(f"\nlabs, mean rate: A {labs['hospital A'].mean():.4f}, B {labs['hospital B'].mean():.4f}, "
          f"A/B ratio median {labs['A / B'].median():.2f} "
          f"(range {labs['A / B'].min():.2f} to {labs['A / B'].max():.2f})")
    print(f"labs, mean rate: septic {labs['septic'].mean():.4f}, non-septic {labs['non-septic'].mean():.4f}")

    gaps = measurement_gaps(grid)
    gaps.to_csv(config.EDA_GAPS_CSV, index=False)
    median_gap = gaps.pivot(index="variable", columns="hospital", values="median_h").reindex(config.MEASURED)
    median_gap.columns = ["hospital A", "hospital B"]
    print("\nmedian hours between measurements:")
    utils.show(median_gap.round(1))

    ks = ks_vitals(grid)
    ks.to_csv(config.EDA_KS_CSV)
    print("\nKS test hospital A vs B on observed vital values:")
    utils.show(ks.round(4))

    grouped_bars(rates, ["hospital A", "hospital B"], "Measurement rate by hospital",
                 "Measurements per patient-hour (log)", "n00_rate_by_hospital.png", log=True)
    grouped_bars(rates, ["septic", "non-septic"], "Measurement rate by sepsis status",
                 "Measurements per patient-hour (log)", "n00_rate_by_sepsis.png", log=True)
    grouped_bars(median_gap, ["hospital A", "hospital B"], "Median hours between measurements",
                 "Hours (log)", "n00_gap_by_hospital.png", log=True)
    grouped_bars(ks.dropna(subset=["KS"]), ["KS"], "KS statistic, hospital A vs B (vitals)",
                 "KS statistic", "n00_ks_vitals.png")
    return rates, gaps, ks


if __name__ == "__main__":
    run()
