#Md Arafat Islam 2111293642
import os

from data_preprocessing import config, utils


def before():
    df = utils.load(config.RAW_PARQUET)
    utils.heading(f"Before: input shape {df.shape}, first 5 rows")
    utils.show(df.head(5))

    duplicated = df.duplicated(keep=False)
    print("duplicated rows:", df.duplicated().sum())
    if duplicated.any():
        utils.heading("examples of duplicated rows:")
        utils.show(df[duplicated].head(5))


def run(force=False):
    df = utils.load(config.RAW_PARQUET)

    if utils.is_done(config.NODUP_PARQUET) and not force:
        print("loaded checkpoint:", os.path.basename(config.NODUP_PARQUET))
        nodup = utils.load(config.NODUP_PARQUET)
    else:
        # patient_id is part of the row, so two identical hours of
        # different patients are not counted as duplicates
        n_duplicates = df.duplicated().sum()
        print(f"fully duplicated rows found: {n_duplicates}")
        nodup = df.drop_duplicates()
        utils.save(nodup, config.NODUP_PARQUET)

    removed = len(df) - len(nodup)
    print("shape before:", df.shape)
    print("shape after: ", nodup.shape)
    if removed == 0:
        print("no duplicated rows, nothing was removed")
    else:
        print(f"removed {removed} duplicated rows")

    utils.heading("After: first 5 rows")
    utils.show(nodup.head(5))
    return nodup


if __name__ == "__main__":
    run()
