#Md Arafat Islam 2111293642
import os
import re
import shutil
import subprocess
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

from data_preprocessing import config, utils

SET_FOLDERS = {"training_setA": "A", "training_setB": "B"}
DOWNLOAD_THREADS = 32


def find_hospital(path):
    """Return 'A' or 'B' if the path is inside training_setA/B, else None."""
    folders = path.replace("\\", "/").split("/")[:-1]
    for name, hospital in SET_FOLDERS.items():
        if name in folders:
            return hospital
    return None


def fetch(url):
    """Download one url and return its bytes, retrying a few times."""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def download_files_into_zip():
    # the folder pages list every patient file, e.g. href="p000001.psv"
    jobs = []
    for folder in SET_FOLDERS:
        folder_url = config.FALLBACK_URL + folder + "/"
        html = fetch(folder_url).decode("utf-8")
        names = sorted(set(re.findall(r'href="(p\d+\.psv)"', html)))
        print(f"{folder}: {len(names)} files listed")
        for name in names:
            jobs.append((folder_url + name, f"{folder}/{name}"))

    # one request per file is slow, so download many files at the same time;
    # every file goes straight into one zip instead of 40k files on disk
    os.makedirs(config.TMP_RAW_DIR, exist_ok=True)
    temp_zip = os.path.join(config.TMP_RAW_DIR, "download.zip.part")
    with zipfile.ZipFile(temp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as pool:
            futures = {}
            for url, arcname in jobs:
                futures[pool.submit(fetch, url)] = arcname
            for future in tqdm(as_completed(futures), total=len(futures), desc="downloading"):
                zf.writestr(futures[future], future.result())
    shutil.move(temp_zip, config.RAW_ZIP)


def download_zip():
    print("downloading the dataset zip ...")
    try:
        subprocess.run(["wget", "-q", "-O", config.RAW_ZIP, config.ZIP_URL], check=True)
    except Exception:
        print("zip download failed (the link may have moved)")

    if os.path.exists(config.RAW_ZIP) and zipfile.is_zipfile(config.RAW_ZIP):
        return
    if os.path.exists(config.RAW_ZIP):
        os.remove(config.RAW_ZIP)

    print("falling back to downloading the 40,336 .psv files one by one (only happens once)")
    download_files_into_zip()
    print("saved", config.RAW_ZIP)


def read_zip(zip_path):
    frames = []
    with zipfile.ZipFile(zip_path) as zf:
        members = []
        for name in zf.namelist():
            hospital = find_hospital(name)
            if name.endswith(".psv") and hospital is not None:
                members.append((name, hospital))
        members.sort()
        print(f"found {len(members)} patient files in the zip")

        for name, hospital in tqdm(members, desc="reading .psv files"):
            with zf.open(name) as f:
                patient = pd.read_csv(f, sep="|")
            patient["patient_id"] = os.path.basename(name).replace(".psv", "")
            patient["hospital"] = hospital
            frames.append(patient)

    df = pd.concat(frames, ignore_index=True)
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
    return df


def get_data(force=False):
    if utils.is_done(config.RAW_PARQUET) and not force:
        print("loaded checkpoint:", os.path.basename(config.RAW_PARQUET))
        return utils.load(config.RAW_PARQUET)

    if not (os.path.exists(config.RAW_ZIP) and zipfile.is_zipfile(config.RAW_ZIP)):
        download_zip()

    start = time.time()
    df = read_zip(config.RAW_ZIP)
    print(f"read all files in {time.time() - start:.1f} seconds")
    utils.save(df, config.RAW_PARQUET)
    return df


def find_demo_in_zip(zf, members):
    """Read patient files in sorted order until the demo patient is found."""
    for name, hospital in members:
        with zf.open(name) as f:
            patient = pd.read_csv(f, sep="|")
        patient["patient_id"] = os.path.basename(name).replace(".psv", "")
        if utils.pick_demo_patient(patient) is not None:
            return name
    return None


def before():
    if not (os.path.exists(config.RAW_ZIP) and zipfile.is_zipfile(config.RAW_ZIP)):
        if utils.is_done(config.RAW_PARQUET):
            print("the raw zip is not in raw/, but sepsis_raw.parquet exists; skipping the raw text preview")
            utils.demo_patient()
        else:
            print("the dataset is not downloaded yet, run() will download it")
        return

    with zipfile.ZipFile(config.RAW_ZIP) as zf:
        members = []
        for name in zf.namelist():
            hospital = find_hospital(name)
            if name.endswith(".psv") and hospital is not None:
                members.append((name, hospital))
        members.sort()

        for folder, hospital in SET_FOLDERS.items():
            names = [name for name, h in members if h == hospital]
            utils.heading(f"{folder}: {len(names)} files, first 5:")
            for name in names[:5]:
                print(" ", name)

        demo_id = utils.DEMO_PATIENT
        if demo_id is None:
            demo_file = find_demo_in_zip(zf, members)
        else:
            demo_file = None
            for name, hospital in members:
                if os.path.basename(name) == demo_id + ".psv":
                    demo_file = name
        if demo_file is None:
            return

        utils.heading(f"Before: first 6 raw lines of {demo_file} (pipe separated, one row per ICU hour)")
        with zf.open(demo_file) as f:
            lines = f.read().decode("utf-8").splitlines()
        for line in lines[:6]:
            print(line)


def run(force=False):
    df = get_data(force)

    patient_id = utils.pick_demo_patient(df)
    utils.heading(f"After: the same patient ({patient_id}) as rows of the table")
    utils.show(utils.demo_rows(df, patient_id, n=5))

    print("\nshape:", df.shape)
    utils.show(df.head())
    print()
    df.info()
    utils.show(df.describe().T.round(2))

    print("\npatients per hospital:")
    patients_per_hospital = df.groupby("hospital")["patient_id"].nunique()
    utils.show(patients_per_hospital.to_frame("patients"))

    row_counts = df["SepsisLabel"].value_counts().sort_index()
    patient_labels = df.groupby("patient_id")["SepsisLabel"].max()
    patient_counts = patient_labels.value_counts().sort_index()

    print("\nSepsisLabel at row level (one row = one ICU hour):")
    utils.show(pd.DataFrame({"rows": row_counts,
                             "percent": (row_counts / row_counts.sum() * 100).round(2)}))
    print("\nSepsisLabel at patient level (patient ever labelled sepsis):")
    utils.show(pd.DataFrame({"patients": patient_counts,
                             "percent": (patient_counts / patient_counts.sum() * 100).round(2)}))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(["No sepsis (0)", "Sepsis (1)"], row_counts.values, color=["tab:blue", "tab:red"])
    axes[0].set_title("Row level (ICU hours)")
    axes[0].set_xlabel("SepsisLabel")
    axes[0].set_ylabel("Number of rows")
    axes[1].bar(["No sepsis (0)", "Sepsis (1)"], patient_counts.values, color=["tab:blue", "tab:red"])
    axes[1].set_title("Patient level")
    axes[1].set_xlabel("SepsisLabel (max over stay)")
    axes[1].set_ylabel("Number of patients")
    utils.finish_plot("Class balance of SepsisLabel", None, None, "s01_class_balance.png")
    return df


if __name__ == "__main__":
    run()
