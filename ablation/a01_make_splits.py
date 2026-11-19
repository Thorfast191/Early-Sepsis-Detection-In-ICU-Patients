import os

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from data_preprocessing import config, utils

# S4 folds: fold 0 trains on hospital A and tests on B, fold 1 the other way round
S4_DIRECTIONS = {0: ("A", "B"), 1: ("B", "A")}


def patient_strata(labels):
    patients = labels.groupby("patient_id").agg(hospital=("hospital", "first"),
                                                 ever_septic=("ever_septic", "max"))
    patients["stratum"] = patients["ever_septic"].astype(str) + "_" + patients["hospital"].astype(str)
    return patients


def split_off(ids, strata, size, seed):
    return train_test_split(ids, test_size=size, stratify=strata, random_state=seed)


def rows(ids_by_role, strategy, seed, fold):
    parts = []
    for role, ids in ids_by_role.items():
        parts.append(pd.DataFrame({"patient_id": list(ids), "strategy": strategy, "seed": seed,
                                   "fold": fold, "role": role}))
    return parts


def make_splits(labels):
    patients = patient_strata(labels)
    ids = patients.index.to_numpy()
    strata = patients["stratum"]
    parts = []

    for seed in config.SPLIT_SEEDS:
        # S1: 70 / 15 / 15
        train, rest = split_off(ids, strata, 0.30, seed)
        val, test = split_off(rest, strata.loc[rest], 0.50, seed)
        parts += rows({"train": train, "val": val, "test": test}, "S1", seed, 0)
        # S2: 80 / 20, no validation set
        train, test = split_off(ids, strata, 0.20, seed)
        parts += rows({"train": train, "test": test}, "S2", seed, 0)

    # S3: K-fold on patients; 10% of each training fold for early stopping
    seed = config.SPLIT_SEEDS[0]
    kfold = StratifiedGroupKFold(n_splits=config.K_FOLDS, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(kfold.split(ids, strata, groups=ids)):
        train_all = ids[train_idx]
        train, val = split_off(train_all, strata.loc[train_all], 0.10, seed)
        parts += rows({"train": train, "val": val, "test": ids[test_idx]}, "S3", seed, fold)

    # S4: external, train on one hospital and test on the other
    for fold, (source, target) in S4_DIRECTIONS.items():
        source_ids = patients.index[patients["hospital"] == source].to_numpy()
        target_ids = patients.index[patients["hospital"] == target].to_numpy()
        train, val = split_off(source_ids, strata.loc[source_ids], 0.10, seed)
        parts += rows({"train": train, "val": val, "test": target_ids}, "S4", seed, fold)

    splits = pd.concat(parts, ignore_index=True)
    check(splits, patients)
    return splits


def split_keys(splits):
    keys = splits[["strategy", "seed", "fold"]].drop_duplicates()
    return [tuple(k) for k in keys.itertuples(index=False)]


def check(splits, patients):
    for (strategy, seed, fold), s in splits.groupby(["strategy", "seed", "fold"]):
        counts = s["patient_id"].value_counts()
        assert counts.max() == 1, f"a patient has more than one role in {strategy} {seed} {fold}"
        if strategy != "S4":
            assert len(s) == len(patients), f"{strategy} {seed} {fold} does not cover every patient"
    s3 = splits[(splits["strategy"] == "S3") & (splits["role"] == "test")]
    assert s3["patient_id"].is_unique and len(s3) == len(patients), "S3 test folds must cover patients once"


def summary(splits, labels):
    patients = patient_strata(labels)
    s = splits.join(patients, on="patient_id")
    table = s.groupby(["strategy", "seed", "fold", "role"]).agg(
        patients=("patient_id", "size"), septic=("ever_septic", "sum"),
        hospital_A=("hospital", lambda h: (h == "A").sum()))
    table["septic %"] = (table["septic"] / table["patients"] * 100).round(2)
    return table


def run(force=False):
    labels = utils.load(config.LABELS_PARQUET)
    if utils.is_done(config.SPLITS_PARQUET) and not force:
        print("loaded checkpoint:", os.path.basename(config.SPLITS_PARQUET))
        splits = utils.load(config.SPLITS_PARQUET)
    else:
        splits = make_splits(labels)
        utils.save(splits, config.SPLITS_PARQUET)
    check(splits, patient_strata(labels))
    print("no patient has more than one role within a split")
    utils.show(summary(splits, labels))
    return splits


if __name__ == "__main__":
    run()
