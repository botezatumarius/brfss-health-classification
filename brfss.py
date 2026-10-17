"""Predicting self-rated general health (GENHLTH) from the BRFSS 2021 survey.

Usage: python brfss.py [stage ...]    Stages run in order; with no argument every stage runs.
Results are written to results/ (CSV) and report/figures/ (PNG).
"""
import sys
import tarfile
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA, RESULTS, FIGURES = ROOT / "data", ROOT / "results", ROOT / "report" / "figures"
SOURCE = "https://www.dlsi.ua.es/~juanra/UA/datasets/"
SEED = 42
SAMPLE = 50_000    # Rows used in the cross-validated experiments (justified by the learning curve).
HOLDOUT = 100_000  # Untouched rows used once, to confirm the final model.
JOBS = 4           # Parallel folds (one per physical core).
CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)  # Same folds in every experiment.
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "bal_acc": "balanced_accuracy", "f1": "f1"}


def download():
    """Download the dataset and the codebook from the course server if they are missing."""
    DATA.mkdir(exist_ok=True)
    for name in ("LLCP2021.csv.tar.gz", "LLCP2021_codebook_llcp-v2-508.pdf"):
        if not (DATA / name).exists():
            urllib.request.urlretrieve(SOURCE + name, DATA / name)
    if not (DATA / "LLCP2021.csv").exists():
        with tarfile.open(DATA / "LLCP2021.csv.tar.gz") as archive:
            archive.extractall(DATA, filter="data")


def load():
    """Raw survey rows with a valid target (GENHLTH 1-5), cached as a pickle for speed."""
    cache = DATA / "llcp2021.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    download()
    raw = pd.read_csv(DATA / "LLCP2021.csv", low_memory=False)
    raw = raw[raw["GENHLTH"].isin([1, 2, 3, 4, 5])].reset_index(drop=True)
    raw.to_pickle(cache)
    return raw


def target(raw, classes=2):
    """Binary: Fair/Poor (4-5) vs the rest. Three classes: Excellent/Very good, Good, Fair/Poor."""
    if classes == 2:
        return (raw["GENHLTH"] >= 4).astype(int)
    return raw["GENHLTH"].map({1: 0, 2: 0, 3: 1, 4: 2, 5: 2}).astype(int)


def splits(raw):
    """Stratified experiment sample and untouched hold-out set (disjoint)."""
    y = target(raw)
    rest, sample = train_test_split(raw.index, test_size=SAMPLE, stratify=y, random_state=SEED)
    _, holdout = train_test_split(rest, test_size=HOLDOUT, stratify=y[rest], random_state=SEED)
    return sample, holdout


STAGES = {}


def main():
    RESULTS.mkdir(exist_ok=True)
    raw = load()
    sample, holdout = splits(raw)
    for name in sys.argv[1:] or STAGES:
        STAGES[name](raw, sample, holdout)


if __name__ == "__main__":
    main()
