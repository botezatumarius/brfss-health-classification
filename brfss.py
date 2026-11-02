"""Predicting self-rated general health (GENHLTH) from the BRFSS 2021 survey.

Usage: python brfss.py [stage ...]    Stages run in order; with no argument every stage runs.
Results are written to results/ (CSV) and report/figures/ (PNG).
"""
import re
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import loguniform, randint, uniform  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

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


def preprocessor(features=None, scale=True):
    """Median imputation with missing indicators for numbers; one-hot (with a 'missing' level) for categories."""
    features = list(FEATURES if features is None else features)
    nominal = [f for f in features if FEATURES.get(f, ("", "num"))[1] == "cat"]
    numeric = [f for f in features if f not in nominal]
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [("impute", SimpleImputer(strategy="constant", fill_value=-1)),
                 ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    return ColumnTransformer([("num", Pipeline(num_steps), numeric), ("cat", Pipeline(cat_steps), nominal)])


def evaluate(model, X, y, name, experiment, extra=None):
    """Stratified 10-fold CV on the shared folds; appends one row per fold to results/<experiment>.csv."""
    start = time.perf_counter()
    scores = cross_validate(model, X, y, cv=CV, scoring=SCORING, n_jobs=JOBS)
    rows = pd.DataFrame({"model": name, "fold": range(CV.n_splits),
                         **{m: scores[f"test_{m}"] for m in SCORING}, "fit_s": scores["fit_time"]})
    for key, value in (extra or {}).items():
        rows[key] = value
    path = RESULTS / f"{experiment}.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old):
        old = old[old["model"] != name]
    pd.concat([old, rows]).to_csv(path, index=False)
    print(f"{experiment:12s} {name:38s} AUC {rows.roc_auc.mean():.4f} ± {rows.roc_auc.std():.4f} "
          f"({time.perf_counter() - start:.0f}s)", flush=True)
    return rows


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


def stage_learning_curve(raw, sample, holdout):
    """AUC on a fixed test set as the training size grows: justifies the experiment sample size."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    X, y = encode(raw), target(raw)
    test = holdout[:50_000]
    pool = raw.index.difference(sample).difference(holdout)
    rows = []
    for size in (2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000):
        train = pd.Index(np.random.default_rng(SEED).choice(pool, size, replace=False))
        for name, clf in (("Logistic regression", LogisticRegression(max_iter=2000)),
                          ("Gradient boosting", HistGradientBoostingClassifier(random_state=SEED))):
            model = Pipeline([("prep", preprocessor()), ("clf", clf)]).fit(X.loc[train], y[train])
            auc = roc_auc_score(y[test], model.predict_proba(X.loc[test])[:, 1])
            rows.append({"model": name, "size": size, "roc_auc": auc})
            print(f"learning curve {name:20s} n={size:7d} AUC {auc:.4f}", flush=True)
    curve = pd.DataFrame(rows)
    curve.to_csv(RESULTS / "learning_curve.csv", index=False)
    plt.figure(figsize=(6, 3.2))
    for i, (name, part) in enumerate(curve.groupby("model")):
        plt.plot(part["size"], part["roc_auc"], marker="o", color=["#eb6834", "#2a78d6"][i], label=name)
    plt.axvline(SAMPLE, color="#52514e", linestyle="--", linewidth=0.8)
    plt.xscale("log")
    plt.gca().set(title="Learning curve (test AUC on 50,000 unseen rows)", xlabel="training rows", ylabel="ROC-AUC")
    plt.legend()
    savefig("learning_curve")


# ------------------------------------------------------------------------------ models
def lightgbm(**params):
    from lightgbm import LGBMClassifier
    base = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.7, random_state=SEED, n_jobs=1, verbose=-1)
    return LGBMClassifier(**{**base, **params})


def classifiers():
    """The compared models, with the settings chosen by manual tuning (stage basic) or common defaults."""
    from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier
    return {
        "Logistic regression": LogisticRegression(C=0.1, max_iter=2000),
        "Decision tree": DecisionTreeClassifier(max_depth=16, min_samples_leaf=100, random_state=SEED),
        "k-nearest neighbours": KNeighborsClassifier(n_neighbors=150),
        "Gaussian naive Bayes": GaussianNB(),
        "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features="sqrt",
                                                random_state=SEED, n_jobs=1),
        "Extra trees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=20, random_state=SEED, n_jobs=1),
        "Gradient boosting (sklearn)": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=SEED),
        "Hist. gradient boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, random_state=SEED),
        "Neural network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                                              random_state=SEED, max_iter=200),
        "LightGBM": lightgbm(),
    }


def data(raw, sample, classes=2):
    return encode(raw.loc[sample]), target(raw.loc[sample], classes)


def pipe(clf, features=None, scale=True, select=None, steps=()):
    parts = [("prep", preprocessor(features, scale))]
    if select is not None:
        parts.append(("select", select))
    return Pipeline([*parts, *steps, ("clf", clf)])


def anova(k):
    from sklearn.feature_selection import SelectKBest, f_classif
    return SelectKBest(f_classif, k=k)


def stage_basic(raw, sample, holdout):
    """Basic level: dummy baseline, filter feature selection and two manually tuned models."""
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    X, y = data(raw, sample)
    for strategy in ("prior", "stratified"):
        evaluate(pipe(DummyClassifier(strategy=strategy, random_state=SEED)), X, y, f"Dummy ({strategy})", "baseline")
    # Filter selection (ANOVA F-test, fitted inside each fold): how many encoded columns are needed?
    for k in (5, 10, 20, 40, 80, "all"):
        evaluate(pipe(LogisticRegression(max_iter=2000), select=anova(k)), X, y, f"LR, top {k} (ANOVA)", "kbest",
                 {"k": k})
    # Manual tuning on the 40 best columns (the curve flattens there).
    for C in (0.001, 0.01, 0.1, 1, 10):
        evaluate(pipe(LogisticRegression(C=C, max_iter=2000), select=anova(40)), X, y, f"LR C={C}",
                 "tuning", {"family": "Logistic regression", "setting": f"C={C}"})
    for depth, leaf in ((3, 1), (5, 1), (8, 1), (12, 1), (None, 1), (8, 100), (12, 100), (12, 400), (16, 100),
                        (16, 200), (20, 200)):
        evaluate(pipe(DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=SEED),
                      select=anova(40)), X, y, f"Tree depth={depth} leaf={leaf}", "tuning",
                 {"family": "Decision tree", "setting": f"max_depth={depth}, min_samples_leaf={leaf}"})


def stage_models(raw, sample, holdout):
    """Intermediate level: ten classifiers on all features, same folds."""
    X, y = data(raw, sample)
    for name, clf in classifiers().items():
        evaluate(pipe(clf), X, y, name, "models")


def stage_imbalance(raw, sample, holdout):
    """Class weights, undersampling, SMOTE and oversampling, applied inside the training folds only."""
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.linear_model import LogisticRegression
    X, y = data(raw, sample)
    for family, make in (("Logistic regression", lambda **kw: LogisticRegression(C=0.1, max_iter=2000, **kw)),
                         ("LightGBM", lambda **kw: lightgbm(**kw))):
        variants = {"none": (make(), None), "class weights": (make(class_weight="balanced"), None),
                    "random undersampling": (make(), RandomUnderSampler(random_state=SEED)),
                    "random oversampling": (make(), RandomOverSampler(random_state=SEED)),
                    "SMOTE": (make(), SMOTE(random_state=SEED))}
        for technique, (clf, sampler) in variants.items():
            steps = [("prep", preprocessor())] + ([("sample", sampler)] if sampler else []) + [("clf", clf)]
            evaluate(ImbPipeline(steps), X, y, f"{family} + {technique}", "imbalance",
                     {"family": family, "technique": technique})


def stage_pca(raw, sample, holdout):
    """Dimensionality reduction: PCA components versus all encoded columns."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    X, y = data(raw, sample)
    full = preprocessor().fit_transform(X)
    ratio = PCA(random_state=SEED).fit(full).explained_variance_ratio_
    pd.DataFrame({"component": range(1, len(ratio) + 1), "explained": ratio}).to_csv(RESULTS / "pca_variance.csv",
                                                                                     index=False)
    plt.figure(figsize=(6, 3))
    plt.plot(range(1, len(ratio) + 1), np.cumsum(ratio), color="#2a78d6")
    for share in (0.8, 0.9):
        plt.axhline(share, color="#52514e", linestyle=":", linewidth=0.8)
    plt.gca().set(title=f"PCA on the {full.shape[1]} encoded columns", xlabel="components",
                  ylabel="cumulative explained variance")
    savefig("pca_variance")
    for n in (5, 10, 20, 40, 80):
        for family, clf in (("Logistic regression", LogisticRegression(C=0.1, max_iter=2000)),
                            ("LightGBM", lightgbm())):
            evaluate(pipe(clf, steps=[("pca", PCA(n_components=n, random_state=SEED))]), X, y,
                     f"{family} + PCA {n}", "pca", {"family": family, "components": n})


def raw_feature(column):
    """Original survey variable behind an encoded column name (e.g. 'cat__EMPLOY1_8.0' -> 'EMPLOY1')."""
    name = column.split("__", 1)[-1].replace("missingindicator_", "")
    for feature in sorted(FEATURES, key=len, reverse=True):
        if name == feature or name.startswith(feature + "_"):
            return feature
    return name


def stage_selection(raw, sample, holdout):
    """Advanced feature selection (fitted inside each fold) and the variable-group hypothesis."""
    from sklearn.feature_selection import RFE, SelectFromModel
    from sklearn.linear_model import LogisticRegression
    X, y = data(raw, sample)
    k = 20
    methods = {
        "Filter: ANOVA F-test": anova(k),
        "Embedded: L1 logistic regression": SelectFromModel(
            LogisticRegression(penalty="l1", C=0.01, solver="liblinear"), max_features=k, threshold=-np.inf),
        "Embedded: LightGBM importance": SelectFromModel(lightgbm(n_estimators=200), max_features=k, threshold=-np.inf),
        "Wrapper: RFE (logistic regression)": RFE(LogisticRegression(C=0.1, max_iter=2000), n_features_to_select=k,
                                                  step=10),
    }
    for name, selector in methods.items():
        evaluate(pipe(lightgbm(), select=selector), X, y, f"{name} (top {k})", "selection", {"method": name, "k": k})
    evaluate(pipe(lightgbm()), X, y, "All features", "selection", {"method": "none", "k": "all"})

    # Which variables do the methods keep? Fitted once on the whole sample, for the report only.
    encoded = preprocessor().fit(X)
    names = encoded.get_feature_names_out()
    Z = encoded.transform(X)
    chosen = {}
    for name, selector in methods.items():
        mask = selector.fit(Z, y).get_support()
        chosen[name] = sorted({raw_feature(c) for c in names[mask]})
    votes = pd.Series([f for kept in chosen.values() for f in kept]).value_counts()
    pd.DataFrame({"votes": votes, "group": [FEATURES[f][0] for f in votes.index]}).to_csv(
        RESULTS / "selection_votes.csv", index_label="feature")

    # Hypothesis: function, chronic disease and socioeconomic status carry almost all the signal.
    groups = pd.Series({f: g for f, (g, _, _) in FEATURES.items()})
    hypothesis = list(groups[groups.isin(["disability", "chronic conditions", "socioeconomic"])].index)
    sets = {f"Only {g}": list(groups[groups == g].index) for g in groups.unique()}
    sets["Hypothesis: disability + chronic + socioeconomic"] = hypothesis
    sets["All except the hypothesis groups"] = [f for f in FEATURES if f not in hypothesis]
    sets["All features"] = list(FEATURES)
    for name, features in sets.items():
        evaluate(pipe(lightgbm(), features=features), X[features], y, name, "groups",
                 {"features": len(features)})


def stage_stats(raw, sample, holdout):
    """Friedman test over all models, then Wilcoxon signed-rank tests (Holm-corrected) on identical folds."""
    from scipy.stats import friedmanchisquare, wilcoxon
    models = pd.read_csv(RESULTS / "models.csv").pivot(index="fold", columns="model", values="roc_auc")
    ranks = models.rank(axis=1, ascending=False).mean().sort_values()
    friedman = friedmanchisquare(*[models[c] for c in models])
    pd.DataFrame({"mean_auc": models.mean(), "std": models.std(), "mean_rank": ranks}).sort_values("mean_rank") \
        .to_csv(RESULTS / "model_ranks.csv", index_label="model")

    def compare(pairs, table):
        rows = []
        for (name_a, a), (name_b, b) in pairs:
            stat, p = wilcoxon(a, b)
            rows.append({"model_a": name_a, "model_b": name_b, "mean_diff": (a - b).mean(),
                         "wins_a": int((a > b).sum()),
                         "statistic": stat, "p": p})
        out = pd.DataFrame(rows).sort_values("p")
        m = len(out)
        out["p_holm"] = np.minimum(1, np.maximum.accumulate(out["p"].values * (m - np.arange(m))))
        out["significant"] = out["p_holm"] < 0.05
        out.to_csv(RESULTS / f"{table}.csv", index=False)
        return out

    top = list(ranks.index[:5])
    compare([((a, models[a]), (b, models[b])) for i, a in enumerate(top) for b in top[i + 1:]], "wilcoxon_models")
    other = []
    groups = pd.read_csv(RESULTS / "groups.csv").pivot(index="fold", columns="model", values="roc_auc")
    other.append((("Hypothesis groups", groups["Hypothesis: disability + chronic + socioeconomic"]),
                  ("All features", groups["All features"])))
    selection = pd.read_csv(RESULTS / "selection.csv").pivot(index="fold", columns="model", values="roc_auc")
    best_selector = selection.drop(columns="All features").mean().idxmax()
    other.append(((best_selector, selection[best_selector]), ("All features", selection["All features"])))
    if (RESULTS / "engineering.csv").exists():
        eng = pd.read_csv(RESULTS / "engineering.csv").pivot(index="fold", columns="model", values="roc_auc")
        for name in ("Logistic regression", "LightGBM"):
            other.append(((f"{name} + engineered", eng[f"{name} + engineered"]), (name, eng[name])))
    if (RESULTS / "nested.csv").exists():
        nested = pd.read_csv(RESULTS / "nested.csv").pivot(index="fold", columns="model", values="outer_auc")
        for name in nested:
            other.append(((f"{name} (tuned, nested CV)", nested[name]), (name, models[name])))
    compare(other, "wilcoxon_other")
    print(f"Friedman chi2={friedman.statistic:.1f}, p={friedman.pvalue:.2e}")
    pd.DataFrame([{"chi2": friedman.statistic, "p": friedman.pvalue, "models": models.shape[1]}]) \
        .to_csv(RESULTS / "friedman.csv", index=False)

    plt.figure(figsize=(7, 3.6))
    order = models.mean().sort_values().index
    plt.boxplot([models[c] for c in order], vert=False, tick_labels=order)
    plt.gca().set(title="ROC-AUC on the 10 shared folds", xlabel="ROC-AUC")
    savefig("models_boxplot")

    summary = groups.agg(["mean", "std"]).T.sort_values("mean")
    plt.figure(figsize=(8, 3.4))
    colors = ["#eb6834" if name.startswith("Hypothesis") else "#2a78d6" for name in summary.index]
    plt.barh(summary.index, summary["mean"], xerr=summary["std"], color=colors)
    plt.xlim(0.5, 0.9)
    plt.gca().set(title="LightGBM on each group of variables", xlabel="ROC-AUC")
    savefig("groups")


# Search space of the automated LightGBM tuning (nested CV and final model).
LIGHTGBM_SPACE = {"clf__n_estimators": randint(200, 1200), "clf__learning_rate": loguniform(0.01, 0.2),
                  "clf__num_leaves": randint(8, 128), "clf__min_child_samples": randint(20, 400),
                  "clf__subsample": uniform(0.5, 0.5), "clf__colsample_bytree": uniform(0.3, 0.7),
                  "clf__reg_lambda": loguniform(1e-3, 30)}


def stage_nested(raw, sample, holdout):
    """Nested cross-validation: randomised search in 3 inner folds, unbiased score on the 10 outer folds."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RandomizedSearchCV
    X, y = data(raw, sample)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    spaces = {
        "Logistic regression": (LogisticRegression(max_iter=3000), {"clf__C": loguniform(1e-3, 10)}, 15),
        "LightGBM": (lightgbm(), LIGHTGBM_SPACE, 25),
    }
    rows = []
    for name, (clf, space, n_iter) in spaces.items():
        for fold, (train, test) in enumerate(CV.split(X, y)):
            search = RandomizedSearchCV(pipe(clf), space, n_iter=n_iter, scoring="roc_auc", cv=inner,
                                        random_state=SEED, n_jobs=JOBS)
            search.fit(X.iloc[train], y.iloc[train])
            auc = roc_auc_score(y.iloc[test], search.predict_proba(X.iloc[test])[:, 1])
            params = {k.replace("clf__", ""): float(round(v, 4)) if isinstance(v, float) else int(v)
                      for k, v in search.best_params_.items()}
            rows.append({"model": name, "fold": fold, "outer_auc": auc, "inner_auc": search.best_score_,
                         "params": params})
            print(f"nested {name:20s} fold {fold} outer {auc:.4f} inner {search.best_score_:.4f} {params}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "nested.csv", index=False)


def engineer(X):
    """Clinically motivated features: WHO BMI classes, guideline thresholds, disease counts and interactions."""
    out = X.copy()
    chronic = ["_RFHYPE6", "_RFCHOL3", "_MICHD", "CVDSTRK3", "CHCSCNCR", "CHCOCNCR", "CHCCOPD3", "ADDEPEV3", "CHCKDNY2",
               "_DRDXAR3"]
    disability = ["DEAF", "BLIND", "DECIDE", "DIFFWALK", "DIFFDRES", "DIFFALON"]
    out["N_CHRONIC"] = X[chronic].sum(axis=1) + (X["DIABETE4"] == 1) + (X["_ASTHMS1"] == 2)
    out["N_DISABILITY"] = X[disability].sum(axis=1)
    bmi = X["_BMI5"]
    out["UNDERWEIGHT"] = (bmi < 18.5).astype(float).where(bmi.notna())
    out["OBESITY_CLASS"] = pd.cut(bmi, [0, 30, 35, 40, np.inf], labels=False, right=False).astype(float)
    limit = np.where(X["_SEX"] == 1, 7, 14)  # CDC heavy drinking: > 7 drinks/week (women), > 14 (men).
    out["HEAVY_DRINKER"] = (X["_DRNKWK1"] > limit).astype(float).where(X["_DRNKWK1"].notna())
    out["FIVE_A_DAY"] = ((X["_FRUTSU1"] + X["_VEGESU1"]) >= 5).astype(float).where(X["_FRUTSU1"].notna())
    out["AGE_X_CHRONIC"] = X["_AGE80"] * out["N_CHRONIC"]
    out["INACTIVE_X_DISABILITY"] = (1 - X["_TOTINDA"]) * out["N_DISABILITY"]
    out["LOW_INCOME_NO_COVER"] = ((X["_INCOMG1"] <= 2) & (X["_HLTHPLN"] == 0)).astype(float)
    out["COST_BARRIER_X_CHRONIC"] = X["MEDCOST1"] * out["N_CHRONIC"]
    return out


def stage_engineering(raw, sample, holdout):
    """Same folds with and without the engineered features, for a linear and a boosted model."""
    from sklearn.linear_model import LogisticRegression
    X, y = data(raw, sample)
    Xe = engineer(X)
    for name, make in (("Logistic regression", lambda: LogisticRegression(C=0.1, max_iter=3000)),
                       ("LightGBM", lightgbm)):
        evaluate(pipe(make()), X, y, name, "engineering")
        evaluate(pipe(make(), features=Xe.columns), Xe, y, f"{name} + engineered", "engineering")


MULTI_SCORING = {"macro_f1": "f1_macro", "macro_auc": "roc_auc_ovr", "weighted_auc": "roc_auc_ovr_weighted",
                 "bal_acc": "balanced_accuracy"}


def stage_multiclass(raw, sample, holdout):
    """Three classes (Excellent/Very good, Good, Fair/Poor) and the original five, with multi-class metrics."""
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import cross_val_predict
    rows = []
    for classes in (3, 5):
        X = encode(raw.loc[sample])
        y = target(raw.loc[sample], 3) if classes == 3 else raw.loc[sample, "GENHLTH"].astype(int) - 1
        models = {"Dummy (stratified)": DummyClassifier(strategy="stratified", random_state=SEED),
                  "Logistic regression": LogisticRegression(C=0.1, max_iter=3000),
                  "Logistic regression (balanced)": LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"),
                  "LightGBM": lightgbm(objective="multiclass"),
                  "LightGBM (balanced)": lightgbm(objective="multiclass", class_weight="balanced")}
        for name, clf in models.items():
            scores = cross_validate(pipe(clf), X, y, cv=CV, scoring=MULTI_SCORING, n_jobs=JOBS)
            rows.append({"classes": classes, "model": name,
                         **{m: scores[f"test_{m}"].mean() for m in MULTI_SCORING},
                         **{f"{m}_std": scores[f"test_{m}"].std() for m in MULTI_SCORING}})
            print(f"multiclass {classes} {name:32s} macro-F1 {rows[-1]['macro_f1']:.4f} "
                  f"macro-AUC {rows[-1]['macro_auc']:.4f}", flush=True)
        if classes == 3:
            pred = cross_val_predict(pipe(lightgbm(objective="multiclass", class_weight="balanced")), X, y, cv=CV,
                                     n_jobs=JOBS)
            cm = confusion_matrix(y, pred, normalize="true")
            plt.figure(figsize=(4.6, 3.6))
            labels = ["Exc./Very good", "Good", "Fair/Poor"]
            sns.heatmap(cm, annot=True, fmt=".0%", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels)
            plt.gca().set(title="3 classes: LightGBM (balanced)", xlabel="predicted", ylabel="true")
            savefig("multiclass_confusion")
    pd.DataFrame(rows).to_csv(RESULTS / "multiclass.csv", index=False)


LABELS = {
    "_AGE80": "Age", "_SEX": "Female", "_RACE": "Race/ethnicity", "MARITAL": "Marital status",
    "_EDUCAG": "Education level", "_INCOMG1": "Income level", "EMPLOY1": "Employment", "RENTHOM1": "Home ownership",
    "VETERAN3": "Veteran", "_CHLDCNT": "Children in household", "_URBSTAT": "Rural county", "_STATE": "State",
    "_HLTHPLN": "Has health insurance", "PERSDOC3": "Personal doctor", "MEDCOST1": "Could not afford doctor",
    "CHECKUP1": "Time since checkup", "_RFHYPE6": "High blood pressure", "_RFCHOL3": "High cholesterol",
    "_MICHD": "Heart attack/CHD", "CVDSTRK3": "Stroke", "_ASTHMS1": "Asthma", "CHCSCNCR": "Skin cancer",
    "CHCOCNCR": "Other cancer", "CHCCOPD3": "COPD", "ADDEPEV3": "Depression", "CHCKDNY2": "Kidney disease",
    "DIABETE4": "Diabetes", "_DRDXAR3": "Arthritis", "_LMTACT3": "Arthritis limits activities", "DEAF": "Deaf",
    "BLIND": "Blind", "DECIDE": "Difficulty concentrating", "DIFFWALK": "Difficulty walking",
    "DIFFDRES": "Difficulty dressing", "DIFFALON": "Difficulty doing errands", "_TOTINDA": "Physically active",
    "_SMOKER3": "Smoking status", "_CURECI1": "E-cigarette user", "USENOW3": "Smokeless tobacco",
    "_RFBING5": "Binge drinker", "_DRNKWK1": "Drinks per week", "_FRUTSU1": "Fruit per day",
    "_VEGESU1": "Vegetables per day", "FRNCHDA_": "Fried potatoes per day", "_BMI5": "BMI", "FLUSHOT7": "Flu shot",
    "PNEUVAC4": "Pneumonia vaccine", "_AIDTST4": "HIV test", "_CHOLCH3": "Cholesterol check",
    "N_CHRONIC": "Number of chronic conditions", "N_DISABILITY": "Number of disabilities",
}


# Codebook meaning of the nominal levels shown in plots.
LEVELS = {
    "_RACE": {1: "White", 2: "Black", 3: "American Indian", 4: "Asian", 5: "Pacific Islander", 6: "other",
              7: "multiracial", 8: "Hispanic"},
    "MARITAL": {1: "married", 2: "divorced", 3: "widowed", 4: "separated", 5: "never married", 6: "unmarried couple"},
    "EMPLOY1": {1: "employed", 2: "self-employed", 3: "out of work 1+ year", 4: "out of work < 1 year",
                5: "homemaker", 6: "student", 7: "retired", 8: "unable to work"},
    "RENTHOM1": {1: "own", 2: "rent", 3: "other"},
    "PERSDOC3": {1: "one", 2: "more than one", 3: "none"},
    "DIABETE4": {1: "yes", 2: "only in pregnancy", 3: "no", 4: "prediabetes"},
    "_LMTACT3": {1: "limited", 2: "not limited", 3: "no arthritis"},
}


def readable(column):
    """Plain-language name of an encoded column, for plots."""
    feature = raw_feature(column)
    name = LABELS.get(feature, feature)
    if "missingindicator_" in column:
        return f"{name} (missing)"
    level = column.split("__", 1)[-1][len(feature):].lstrip("_")
    if not level:
        return name
    code = int(float(level))
    return f"{name}: {'missing' if code == -1 else LEVELS.get(feature, {}).get(code, code)}"


def shap_values(clf, Z):
    """SHAP values of the positive class (older SHAP versions return one array per class)."""
    import shap
    values = shap.TreeExplainer(clf).shap_values(Z)
    return values[1] if isinstance(values, list) else values


def youden(y, p):
    """Threshold maximising sensitivity + specificity - 1."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y, p)
    return thresholds[np.argmax(tpr - fpr)]


def stage_errors(raw, sample, holdout):
    """Instances that the three best models all misclassify, analysed with out-of-fold SHAP values."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    X, y = data(raw, sample)
    models = {"LightGBM": lightgbm(),
              "Hist. gradient boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                                                        random_state=SEED),
              "Logistic regression": LogisticRegression(C=0.1, max_iter=2000)}
    probs, wrong = {}, []
    for name, clf in models.items():
        p = cross_val_predict(pipe(clf), X, y, cv=CV, method="predict_proba", n_jobs=JOBS)[:, 1]
        probs[name] = p
        wrong.append((p >= youden(y, p)).astype(int) != y.values)
    always = np.logical_and.reduce(wrong)
    mean_p = pd.Series(np.mean(list(probs.values()), axis=0), index=X.index)
    summary = pd.DataFrame({"true": y, "mean_p": mean_p, "always_wrong": always})
    summary.to_csv(RESULTS / "oof_predictions.csv", index_label="row")

    # The two most confident false negatives and false positives shared by all three models.
    fn = summary[(summary.true == 1) & summary.always_wrong].nsmallest(2, "mean_p").index
    fp = summary[(summary.true == 0) & summary.always_wrong].nlargest(2, "mean_p").index
    cases = list(fn) + list(fp)
    Xe = engineer(X)
    show = ["_AGE80", "_SEX", "_INCOMG1", "_EDUCAG", "EMPLOY1", "N_CHRONIC", "N_DISABILITY", "DIFFWALK", "ADDEPEV3",
            "_TOTINDA", "_SMOKER3", "_BMI5"]
    profile = Xe.loc[cases, show].T
    profile.columns = [f"{'FN' if summary.true[c] else 'FP'} #{c}" for c in cases]
    profile["Fair/Poor (median)"] = Xe.loc[y == 1, show].median()
    profile["Exc./VG/Good (median)"] = Xe.loc[y == 0, show].median()
    profile.index = [LABELS.get(f, f) for f in show]
    profile.loc["P(Fair/Poor), mean of 3 models"] = list(mean_p[cases].round(3)) + [np.nan, np.nan]
    profile.to_csv(RESULTS / "error_cases.csv")

    # Groups behind the errors: consistent false negatives vs correctly found Fair/Poor respondents.
    groups = {"False negatives": (y == 1) & always, "Other Fair/Poor": (y == 1) & ~always,
              "False positives": (y == 0) & always, "Other Exc./VG/Good": (y == 0) & ~always}
    ordered = [f for f in show if FEATURES.get(f, ("", "num"))[1] != "cat"]
    table = pd.DataFrame({name: Xe.loc[mask, ordered].mean() for name, mask in groups.items()}).rename(index=LABELS)
    table.loc["People"] = [mask.sum() for mask in groups.values()]
    table.to_csv(RESULTS / "error_groups.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for value, color, label in ((0, "#2a78d6", CLASS_NAMES[0]), (1, "#eb6834", CLASS_NAMES[1])):
        axes[0].hist(mean_p[y == value], bins=40, alpha=0.6, color=color, label=label, density=True)
    for c in cases:
        axes[0].axvline(mean_p[c], color="#0b0b0b", linestyle=":", linewidth=1)
    axes[0].set(title="Out-of-fold P(Fair/Poor)", xlabel="mean probability of the 3 models")
    axes[0].legend(fontsize=7)
    burden = Xe["N_CHRONIC"] + Xe["N_DISABILITY"]
    rate = pd.DataFrame({"burden": burden.clip(upper=8), "fn": groups["False negatives"], "pos": y == 1})
    share = rate[rate.pos].groupby("burden")["fn"].mean()
    axes[1].bar(share.index, share.values, color="#eb6834")
    axes[1].set(title="Fair/Poor respondents missed by all 3 models", xlabel="chronic conditions + disabilities",
                ylabel="share missed")
    savefig("errors_overview")

    # SHAP waterfall of each case, from the LightGBM model of the fold in which it was held out.
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    for ax, case in zip(axes.flat, cases):
        position = X.index.get_loc(case)
        train = next(tr for tr, te in CV.split(X, y) if position in te)
        model = pipe(lightgbm()).fit(X.iloc[train], y.iloc[train])
        row = model["prep"].transform(X.loc[[case]])
        values = shap_values(model["clf"], row)[0]
        names = [readable(c) for c in model["prep"].get_feature_names_out()]
        order = np.argsort(np.abs(values))[::-1][:8][::-1]
        colors = ["#eb6834" if v > 0 else "#2a78d6" for v in values[order]]
        ax.barh([names[i] for i in order], values[order], color=colors)
        kind = "false negative" if y[case] else "false positive"
        ax.set_title(f"Respondent #{case}: {kind}, P = {mean_p[case]:.2f}", fontsize=9)
        ax.set_xlabel("SHAP contribution (log-odds)", fontsize=8)
        ax.axvline(0, color="#52514e", linewidth=0.8)
        ax.tick_params(axis="y", labelsize=8)
    savefig("errors_shap")
    print("consistently misclassified:", {k: int(v.sum()) for k, v in groups.items()}, "cases", cases)


def stage_xai(raw, sample, holdout):
    """SHAP explanations of LightGBM: global importance, feature groups and dependence plots."""
    import shap
    X, y = data(raw, sample)
    model = pipe(lightgbm()).fit(X, y)
    rows = X.sample(3000, random_state=SEED)
    Z = model["prep"].transform(rows)
    names = model["prep"].get_feature_names_out()
    values = shap_values(model["clf"], Z)
    importance = pd.Series(np.abs(values).mean(axis=0), index=names)
    by_feature = importance.groupby([raw_feature(n) for n in names]).sum().sort_values(ascending=False)
    by_group = by_feature.groupby([FEATURES[f][0] for f in by_feature.index]).sum().sort_values()
    by_feature.rename(index=LABELS).to_csv(RESULTS / "shap_features.csv", header=["mean_abs_shap"])
    by_group.to_csv(RESULTS / "shap_groups.csv", header=["mean_abs_shap"])

    plt.figure()
    shap.summary_plot(values, Z, feature_names=[readable(n) for n in names], max_display=15, show=False,
                      plot_size=(7, 5))
    plt.title("SHAP values of the 15 most influential columns (LightGBM)")
    savefig("shap_beeswarm")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    top = by_feature.head(12)[::-1]
    axes[0].barh([LABELS.get(f, f) for f in top.index], top.values, color="#2a78d6")
    axes[0].set(title="Importance per survey variable", xlabel="mean |SHAP|")
    axes[1].barh(by_group.index, by_group.values, color="#1baf7a")
    axes[1].set(title="Importance per variable group", xlabel="mean |SHAP|")
    savefig("shap_importance")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, feature in zip(axes, ("_AGE80", "_BMI5")):
        column = list(names).index(f"num__{feature}")
        raw_values = rows[feature]
        ax.scatter(raw_values, values[:, column], s=4, alpha=0.4, color="#2a78d6")
        ax.axhline(0, color="#52514e", linewidth=0.8)
        ax.set(title=f"{LABELS[feature]}: effect on the prediction", xlabel=LABELS[feature], ylabel="SHAP value")
    savefig("shap_dependence")


STAGES = {"eda": stage_eda,
          "curve": stage_learning_curve,
          "basic": stage_basic,
          "models": stage_models,
          "imbalance": stage_imbalance,
          "pca": stage_pca,
          "selection": stage_selection,
          "engineering": stage_engineering,
          "stats": stage_stats,
          "nested": stage_nested,
          "multiclass": stage_multiclass,
          "errors": stage_errors,
          "xai": stage_xai}


def main():
    RESULTS.mkdir(exist_ok=True)
    raw = load()
    sample, holdout = splits(raw)
    for name in sys.argv[1:] or STAGES:
        STAGES[name](raw, sample, holdout)


if __name__ == "__main__":
    main()
