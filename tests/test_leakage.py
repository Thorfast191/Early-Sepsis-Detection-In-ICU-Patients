import numpy as np

from ablation import a01_make_splits, a02_train_eval
from novel_preprocessing import n07_fitted_transforms as n07

FITS_PER_ARM = {"BASE": 2, "NOVEL": 4}


def run():
    data = a02_train_eval.load_data()
    splits = n07.load_splits()
    labels = data["labels"]
    for key in a01_make_splits.split_keys(splits):
        roles = a02_train_eval.split_roles(labels, key)
        n07.FIT_LOG.clear()
        base = a02_train_eval.base_matrix(data, roles, key)
        a02_train_eval.novel_matrix(data, roles, key, base, True)
        a02_train_eval.novel_matrix(data, roles, key, base, False)
        assert len(n07.FIT_LOG) == FITS_PER_ARM["BASE"] + 2 * FITS_PER_ARM["NOVEL"], n07.FIT_LOG
        assert all(f["n_outside"] == 0 for f in n07.FIT_LOG), n07.FIT_LOG
        n_train = len(n07.train_ids(key))
        assert all(f["n_patients"] == n_train for f in n07.FIT_LOG), "a fit did not see all training patients"
        print(f"{key}: {len(n07.FIT_LOG)} fits, every one saw only the {n_train} training patients")

    # negative control: a fit that receives a test patient must fail
    key = a01_make_splits.split_keys(splits)[0]
    roles = a02_train_eval.split_roles(labels, key)
    rows = (roles == "train") | (roles == "test")
    try:
        n07.fit_medians(data["base"][rows], labels["patient_id"].to_numpy()[rows], key)
    except AssertionError:
        print("negative control OK: fitting on test patients raises")
    else:
        raise AssertionError("fit on test patients was not caught")
    assert np.all(roles[roles != "train"] != "train")


if __name__ == "__main__":
    run()
