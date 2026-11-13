import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from data_preprocessing import config, utils

AGE_BINS = [0, 40, 60, 80, float("inf")]
# plain names: some libraries (e.g. XGBoost) reject '<' in column names
AGE_LABELS = ["under40", "40to60", "60to80", "80plus"]


def encode(df, age_group, hospital_encoder):
    df = df.copy()
    df["hospital"] = hospital_encoder.transform(df["hospital"])

    icu_dummies = pd.get_dummies(df["ICU_type"], prefix="ICU", dtype=int)
    df = pd.concat([df.drop(columns=["ICU_type"]), icu_dummies], axis=1)

    age_dummies = pd.get_dummies(age_group, prefix="Age", dtype=int)
    df = pd.concat([df, age_dummies], axis=1)
    return df


def run(force=False):
    clean = utils.load(config.CLEAN_PARQUET)

    hospital_encoder = LabelEncoder()
    hospital_encoder.fit(clean["hospital"])
    mapping = {}
    for i, name in enumerate(hospital_encoder.classes_):
        mapping[name] = i

    # age groups come from the unscaled Age in the clean data
    age_group = pd.cut(clean["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)

    done = utils.is_done(config.FINAL_PARQUET) and utils.is_done(config.FINAL_UNSCALED_PARQUET)
    if done and not force:
        print("loaded checkpoint:", os.path.basename(config.FINAL_PARQUET))
        print("loaded checkpoint:", os.path.basename(config.FINAL_UNSCALED_PARQUET))
        final = utils.load(config.FINAL_PARQUET)
        final_unscaled = utils.load(config.FINAL_UNSCALED_PARQUET)
    else:
        scaled = utils.load(config.SCALED_PARQUET)
        # both files come from s04 and keep its row order and index
        assert scaled.index.equals(clean.index), "scaled and clean rows do not line up"
        final = encode(scaled, age_group, hospital_encoder)
        final_unscaled = encode(clean, age_group, hospital_encoder)
        utils.save(final, config.FINAL_PARQUET)
        utils.save(final_unscaled, config.FINAL_UNSCALED_PARQUET)

    print("label encoding of hospital:", mapping)
    print("one-hot encoding of ICU_type:", sorted(clean["ICU_type"].unique()))
    print("age groups (from unscaled Age):")
    utils.show(age_group.value_counts().reindex(AGE_LABELS).to_frame("rows"))
    print("Gender is already 0/1 in the raw data, so it needs no encoding:",
          sorted(clean["Gender"].unique().tolist()))

    new_cols = ["hospital"] + [c for c in final.columns if c.startswith("ICU_") or c.startswith("Age_")]
    print("\nnew encoded columns:")
    utils.show(final[["patient_id", "Age"] + new_cols].head())

    print("final shape (scaled + encoded):  ", final.shape)
    print("final shape (unscaled + encoded):", final_unscaled.shape)
    return final


if __name__ == "__main__":
    run()
