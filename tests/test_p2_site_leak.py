import numpy as np

from ablation import a02_train_eval
from novel_preprocessing import n07_fitted_transforms as n07
from novel_preprocessing import n09_site_screening as n09

VARIABLES = ["HR", "MAP", "Temp", "Lactate", "WBC"]
TOLERANCE = 0.05


def run():
    """P2 must not make the hospital easier to identify than the raw values do."""
    data = a02_train_eval.load_data()
    labels = data["labels"]
    key = ("S1", 42, 0)
    train = a02_train_eval.split_roles(labels, key) == "train"
    ids = labels["patient_id"].to_numpy()[train]
    hospitals = labels["hospital"].astype(str).to_numpy()[train]
    p1 = data["p1"][train]
    cols = [f"{v}_{kind}" for v in VARIABLES for kind in ["hsince", "rate", "last"]]
    refs = n07.fit_p2(p1[cols], ids, hospitals, key)
    encoded = n07.apply_p2(p1[cols], hospitals, refs, n07.row_keys(ids, labels["ICULOS"].to_numpy()[train]))

    rows = n09.balanced_sample(hospitals, 30_000, np.random.default_rng(0))
    is_a = (hospitals[rows] == "A").astype(int)
    for v in VARIABLES:
        vc = [c for c in cols if c.startswith(v + "_")]
        raw = n09.domain_auroc(p1[vc].to_numpy("float32")[rows], is_a, ids[rows], np.random.default_rng(1))
        p2 = n09.domain_auroc(encoded[vc].to_numpy()[rows], is_a, ids[rows], np.random.default_rng(1))
        print(f"{v}: hospital AUROC raw {raw:.3f}, after P2 {p2:.3f}")
        assert p2 <= raw + TOLERANCE, f"P2 makes the hospital of {v} easier to identify"
    print("P2 site-leak check OK")


if __name__ == "__main__":
    run()
