import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ablation import a02_train_eval, a03_stats
from data_preprocessing import config, utils


def run():
    labels = utils.load(config.LABELS_PARQUET)
    index = a03_stats.prediction_index()
    n_sets = 0
    for key, jobs in index.groupby(a03_stats.GROUP_COLS):
        strategy, seed, fold, horizon, model = key
        frames = {row.arm: pd.read_parquet(row.path) for row in jobs.itertuples()}
        assert {"BASE", "NOVEL"} <= set(frames), f"missing arm in {key}"
        ref = frames["BASE"][["patient_id", "ICULOS", "y"]].reset_index(drop=True)
        for arm, frame in frames.items():
            same = frame[["patient_id", "ICULOS", "y"]].reset_index(drop=True)
            assert same.equals(ref), f"{arm} test rows differ from BASE in {key}"
        # and the labels are the n01 labels of those rows
        merged = ref.merge(labels[["patient_id", "ICULOS", f"label_{horizon}"]], on=["patient_id", "ICULOS"])
        assert len(merged) == len(ref) and (merged["y"] == merged[f"label_{horizon}"]).all()
        n_sets += 1
    print(f"identical evaluation sets OK: {n_sets} test sets, all arms share (patient_id, ICULOS, label)")

    # the fast bootstrap metrics give the sklearn values with unit weights
    pred = pd.read_parquet(index["path"].iloc[0])
    y, p = pred["y"].to_numpy(), pred["p"].to_numpy(dtype="float64")
    m = a03_stats.SortedScores(y, p).metrics(np.ones((1, len(y))))
    assert np.isclose(m["auroc"][0], roc_auc_score(y, p))
    assert np.isclose(m["auprc"][0], average_precision_score(y, p))
    assert np.isclose(m["brier"][0], brier_score_loss(y, p))
    print("weighted AUROC / AUPRC / Brier match sklearn")

    # docstring example of the official evaluate_sepsis_score.py
    u = a02_train_eval.compute_prediction_utility([0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 1, 1])
    assert np.isclose(u, 3.388888888888889)
    print("utility matches the reference value 3.388888888888889")


if __name__ == "__main__":
    run()
