"""Predicting self-rated general health (GENHLTH) from the BRFSS 2021 survey.

Usage: python brfss.py [stage ...]    Stages run in order; with no argument every stage runs.
Results are written to results/ (CSV) and report/figures/ (PNG).
"""
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
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

# Feature specification, from the codebook: name -> (group, kind, details).
#   num: (divisor, missing codes)        bin: (code meaning "yes", missing codes)
#   ord: {code: rank} (other codes = missing)   cat: valid codes (other codes = missing)
DK = (7, 9)
FEATURES = {
    "_AGE80": ("demographics", "num", (1, ())),
    "_SEX": ("demographics", "bin", (2, ())),  # 1 = female
    "_RACE": ("demographics", "cat", (1, 2, 3, 4, 5, 6, 7, 8)),
    "MARITAL": ("demographics", "cat", (1, 2, 3, 4, 5, 6)),
    "_EDUCAG": ("socioeconomic", "ord", {1: 1, 2: 2, 3: 3, 4: 4}),
    "_INCOMG1": ("socioeconomic", "ord", {k: k for k in range(1, 8)}),
    "EMPLOY1": ("socioeconomic", "cat", (1, 2, 3, 4, 5, 6, 7, 8)),
    "RENTHOM1": ("socioeconomic", "cat", (1, 2, 3)),
    "VETERAN3": ("demographics", "bin", (1, DK)),
    "_CHLDCNT": ("demographics", "ord", {k: k - 1 for k in range(1, 7)}),
    "_URBSTAT": ("demographics", "bin", (2, ())),  # 1 = rural
    "_STATE": ("demographics", "cat", None),  # Every state code is valid.
    "_HLTHPLN": ("health care", "bin", (1, (9,))),
    "PERSDOC3": ("health care", "cat", (1, 2, 3)),
    "MEDCOST1": ("health care", "bin", (1, DK)),
    "CHECKUP1": ("health care", "ord", {1: 1, 2: 2, 3: 3, 4: 4, 8: 5}),
    "_RFHYPE6": ("chronic conditions", "bin", (2, (9,))),
    "_RFCHOL3": ("chronic conditions", "bin", (2, (9,))),
    "_MICHD": ("chronic conditions", "bin", (1, ())),
    "CVDSTRK3": ("chronic conditions", "bin", (1, DK)),
    "_ASTHMS1": ("chronic conditions", "ord", {3: 0, 2: 1, 1: 2}),
    "CHCSCNCR": ("chronic conditions", "bin", (1, DK)),
    "CHCOCNCR": ("chronic conditions", "bin", (1, DK)),
    "CHCCOPD3": ("chronic conditions", "bin", (1, DK)),
    "ADDEPEV3": ("chronic conditions", "bin", (1, DK)),
    "CHCKDNY2": ("chronic conditions", "bin", (1, DK)),
    "DIABETE4": ("chronic conditions", "cat", (1, 2, 3, 4)),
    "_DRDXAR3": ("chronic conditions", "bin", (1, ())),
    "_LMTACT3": ("disability", "cat", (1, 2, 3)),
    "DEAF": ("disability", "bin", (1, DK)),
    "BLIND": ("disability", "bin", (1, DK)),
    "DECIDE": ("disability", "bin", (1, DK)),
    "DIFFWALK": ("disability", "bin", (1, DK)),
    "DIFFDRES": ("disability", "bin", (1, DK)),
    "DIFFALON": ("disability", "bin", (1, DK)),
    "_TOTINDA": ("lifestyle", "bin", (1, (9,))),  # 1 = physically active
    "_SMOKER3": ("lifestyle", "ord", {4: 0, 3: 1, 2: 2, 1: 3}),
    "_CURECI1": ("lifestyle", "bin", (2, (9,))),
    "USENOW3": ("lifestyle", "ord", {3: 0, 2: 1, 1: 2}),
    "_RFBING5": ("lifestyle", "bin", (2, (9,))),
    "_DRNKWK1": ("lifestyle", "num", (100, (99900,))),
    "_FRUTSU1": ("lifestyle", "num", (100, ())),
    "_VEGESU1": ("lifestyle", "num", (100, ())),
    "FRNCHDA_": ("lifestyle", "num", (100, ())),
    "_BMI5": ("lifestyle", "num", (100, ())),
    "FLUSHOT7": ("prevention", "bin", (1, DK)),
    "PNEUVAC4": ("prevention", "bin", (1, DK)),
    "_AIDTST4": ("prevention", "bin", (1, (9,))),
    "_CHOLCH3": ("prevention", "ord", {1: 0, 2: 1, 3: 2}),
}
# Measures of the same construct as the target (CDC HRQOL-4 "healthy days"), or computed from it.
LEAKAGE = ["_RFHLTH", "PHYSHLTH", "MENTHLTH", "POORHLTH", "_PHYS14D", "_MENT14D"]
CLASS_NAMES = {0: "Excellent/Very good/Good", 1: "Fair/Poor"}


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


def encode(raw):
    """Turn the survey codes into model-ready values: unknown/refused/missing codes become NaN."""
    out = pd.DataFrame(index=raw.index)
    for name, (_, kind, spec) in FEATURES.items():
        col = raw[name]
        if kind == "num":
            divisor, missing = spec
            out[name] = col.where(~col.isin(missing)) / divisor
        elif kind == "bin":
            positive, missing = spec
            valid = col.notna() & ~col.isin(missing)
            out[name] = (col == positive).astype(float).where(valid)
        elif kind == "ord":
            out[name] = col.map(spec)
        else:
            out[name] = col if spec is None else col.where(col.isin(spec))
    return out


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


def savefig(name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}.png", dpi=150)
    plt.close()


def codebook_missing_codes():
    """Codes the codebook labels as don't know / refused / missing, for every variable."""
    import pypdf
    pages = pypdf.PdfReader(DATA / "LLCP2021_codebook_llcp-v2-508.pdf").pages
    text = "\n".join(page.extract_text() or "" for page in pages)
    codes = {}
    for entry in re.split(r"\nLabel: ", text)[1:]:
        name = re.search(r"SAS Variable Name: (\S+)", entry)
        if name:
            found = re.findall(r"\n(\d+) [^\n]*(?:Don.t know|Not sure|Refused|Missing)", entry, flags=re.I)
            codes.setdefault(name.group(1), {int(c) for c in found})
    return codes


# ----------------------------------------------------------------------------------- EDA
def stage_eda(raw, sample, holdout):
    """Target imbalance, missing values, leakage screening and associations with the target."""
    counts = raw["GENHLTH"].value_counts().sort_index()
    labels = ["Excellent", "Very good", "Good", "Fair", "Poor"]
    pd.DataFrame({"class": labels, "count": counts.values, "share": (counts / counts.sum()).round(4).values}) \
        .to_csv(RESULTS / "target_distribution.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].bar(labels, counts.values, color=["#2a78d6"] * 3 + ["#eb6834"] * 2)
    axes[0].set(title="GENHLTH (5 classes)", ylabel="respondents")
    binary = target(raw).value_counts().sort_index()
    axes[1].bar([CLASS_NAMES[0], CLASS_NAMES[1]], binary.values, color=["#2a78d6", "#eb6834"])
    axes[1].set(title=f"Binary target ({binary[1] / binary.sum():.1%} positive)")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    savefig("target")

    # Missing values of every column once the codebook's unknown/refused codes are treated as missing.
    codes = codebook_missing_codes()
    missing = pd.Series({c: raw[c].where(~raw[c].isin(codes.get(c, set()))).isna().mean() for c in raw.columns})
    missing.sort_values().to_csv(RESULTS / "missing_all_columns.csv", header=["missing"])
    plt.figure(figsize=(6, 3))
    plt.hist(missing, bins=20, color="#2a78d6")
    plt.axvline(0.5, color="#eb6834", linestyle="--")
    plt.gca().set(title="Missing values per column (303 columns)", xlabel="share missing", ylabel="columns")
    savefig("missing_all")

    # Leakage screening: AUC of each column on its own (missing codes -> median).
    y = target(raw)
    screen = {}
    for c in raw.columns.drop("GENHLTH"):
        x = pd.to_numeric(raw[c].where(~raw[c].isin(codes.get(c, set()))), errors="coerce")
        if x.notna().mean() > 0.5 and x.nunique() > 1:
            auc = roc_auc_score(y, x.fillna(x.median()))
            screen[c] = max(auc, 1 - auc)
    screen = pd.Series(screen).sort_values(ascending=False)
    screen.to_csv(RESULTS / "single_feature_auc.csv", header=["auc"])
    top = screen.head(15)[::-1]
    plt.figure(figsize=(6, 4))
    plt.barh(top.index, top.values, color=["#eb6834" if c in LEAKAGE else "#2a78d6" for c in top.index])
    plt.gca().set(title="Single-column AUC (orange: removed as leakage)", xlabel="ROC-AUC", xlim=(0.5, 1))
    savefig("leakage")

    # Missing values and target association of the selected features.
    X = encode(raw)
    info = pd.DataFrame({"group": [FEATURES[f][0] for f in X], "kind": [FEATURES[f][1] for f in X],
                         "missing": X.isna().mean().round(3), "auc_alone": screen.reindex(X.columns).round(3)})
    info.to_csv(RESULTS / "features.csv", index_label="feature")
    rates = []
    for f in ["DIFFWALK", "CHCCOPD3", "_DRDXAR3", "ADDEPEV3", "_TOTINDA", "_SMOKER3", "_INCOMG1", "_EDUCAG"]:
        rates.append(y.groupby(X[f]).mean().rename(f))
    rates = pd.concat(rates, axis=1).T
    rates.to_csv(RESULTS / "fair_poor_rate_by_feature.csv")

    plt.figure(figsize=(6, 3.2))
    age = pd.cut(X["_AGE80"], [17, 34, 44, 54, 64, 74, 80], labels=["18-34", "35-44", "45-54", "55-64", "65-74", "75+"])
    incomes = ["<15k", "15-25k", "25-35k", "35-50k", "50-100k", "100-200k", ">200k"]
    inc = X["_INCOMG1"].map(dict(zip(range(1, 8), incomes)))
    sns.heatmap(pd.crosstab(inc, age, values=y, aggfunc="mean").reindex(incomes),
                annot=True, fmt=".0%", cmap="Oranges", cbar=False)
    plt.gca().set(title="Share in Fair/Poor health by income and age", xlabel="age", ylabel="household income ($)")
    savefig("income_age")

    ordered = [f for f in X if FEATURES[f][1] != "cat"]  # Correlation is meaningless for nominal codes.
    corr = X[ordered].corr(method="spearman")
    plt.figure(figsize=(9, 7.5))
    sns.heatmap(corr, cmap="vlag", center=0, xticklabels=True, yticklabels=True)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.title("Spearman correlation between the numeric, binary and ordinal features")
    savefig("correlation")
    print(f"target: {binary[1] / binary.sum():.3f} positive; columns with >50% missing: {(missing > 0.5).sum()}")
    print("top single-column AUC:", screen.head(8).round(3).to_dict())


STAGES = {"eda": stage_eda}


def main():
    RESULTS.mkdir(exist_ok=True)
    raw = load()
    sample, holdout = splits(raw)
    for name in sys.argv[1:] or STAGES:
        STAGES[name](raw, sample, holdout)


if __name__ == "__main__":
    main()
