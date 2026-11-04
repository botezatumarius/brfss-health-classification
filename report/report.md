# Predicting Self-Rated Health from the BRFSS 2021 Survey

**Author:** Marius Botezatu  
**Date:** 7 November 2026  
**Code:** `brfss.py` — <https://github.com/botezatumarius/brfss-health-classification>

## 1. Introduction

The Behavioral Risk Factor Surveillance System (BRFSS) is the largest continuous health survey in the
world: every year it interviews more than 400,000 adults in the United States about their health,
habits and living conditions. One of its first questions is simple: *"Would you say that in general your
health is excellent, very good, good, fair or poor?"* (`GENHLTH`). This self-rating is one of the best
single predictors of future illness and mortality, so understanding what drives it is useful for
public health.

**Goal.** Predict whether a person rates their health as **fair or poor** from information about their
chronic conditions, disabilities, lifestyle, use of health care and socioeconomic situation, and find
which of these factors matter most.

**Approach in brief.** The five answers are grouped into a binary problem: *Fair/Poor* (16.6 % of people)
against *Excellent/Very good/Good*. The data is cleaned with the official codebook, leaking variables are
removed, and ten classifiers are compared with the same stratified 10-fold cross-validation. The
analysis then studies class imbalance, dimensionality reduction, feature selection, statistical
significance, errors and explanations (SHAP), and finally tunes the best model with nested
cross-validation, adds clinically motivated features, solves the three-class version of the problem,
and confirms the result on 100,000 people never used before.

| Level of the guidelines | Where |
|---|---|
| Basic: cleaning, EDA, dummy baseline, feature selection, 2 tuned models, 10-fold CV | Sections 2–5 |
| Intermediate: 10 models, imbalance, PCA, advanced selection + hypothesis, Wilcoxon, errors, SHAP | Section 6 |
| Advanced: nested CV with automatic search, LightGBM, three classes, feature engineering | Section 7 |
| Final model, recommendation and conclusions | Sections 8–9 |
| Ethics | Section 10 |

## 2. Data and preprocessing

### 2.1 The dataset

The 2021 BRFSS file has **438,693 interviews and 303 columns**. Each column is a survey answer or a
variable computed by the CDC, and every code is documented in the codebook. Three problems stand out:

- **Special codes instead of empty cells.** "Don't know", "Refused" and "Not asked" are stored as numbers
  (7/9, 77/99, 777/999…), and some numbers have other meanings: in `PHYSHLTH`, 88 means "zero days".
  Used as they are, a model would read "refused" as "a lot".
- **Many columns are mostly empty.** Optional modules are asked only in some states: once the special
  codes are treated as missing, **160 of the 303 columns are more than 50 % empty** (Figure 1).
- **Administrative columns.** Interview dates, phone-screening questions, survey weights and record IDs
  describe the interview, not the person.

![Missing values](figures/missing_all.png)

*Figure 1: share of missing values per column, after the codebook's "don't know / refused / not asked"
codes are converted to missing. The dashed line marks 50 %.*

The target has five classes, and 1,161 interviews with "don't know", "refused" or no answer were removed,
leaving **437,532 people**.

### 2.2 Leakage: variables that already contain the answer

The guidelines warn that some variables are directly related to the target. To find them objectively,
every column was used *alone* to predict the target and its ROC-AUC was measured (Figure 2):

![Leakage screening](figures/leakage.png)

*Figure 2: ROC-AUC of each column on its own (top 15). Orange columns were removed.*

- **`_RFHLTH`** reaches an AUC of exactly **1.000**: the codebook confirms it is *computed from* `GENHLTH`
  ("adults with good or better health"). It is the target itself.
- **`PHYSHLTH`, `MENTHLTH`, `POORHLTH`** ("days of poor physical/mental health") and their derived
  versions **`_PHYS14D`, `_MENT14D`** belong to the CDC's *Healthy Days* measure (HRQOL-4), which asks
  the same self-assessment as `GENHLTH` in another form. `_PHYS14D` alone reaches 0.763, more than any
  real risk factor.

All six were removed: the aim is to predict self-rated health from *causes and circumstances*, not from
another version of the same self-rating.

### 2.3 Selected variables

From the columns that are less than half empty, after dropping administrative and leaking ones, **49
variables** were kept in seven meaningful groups. When the CDC provides a cleaned "calculated" version of
an answer (for example `_BMI5` instead of raw weight and height, or `_SMOKER3` instead of three smoking
questions) only that version is kept, to avoid near-duplicate columns.

| Group | Variables (examples) | Count |
|---|---|---|
| Demographics | age, sex, race/ethnicity, marital status, state, rural county | 8 |
| Socioeconomic | income, education, employment, home ownership | 4 |
| Health care | insurance, personal doctor, could not afford a doctor, last checkup | 4 |
| Chronic conditions | blood pressure, cholesterol, heart disease, stroke, COPD, diabetes, depression, kidney disease, cancer, asthma, arthritis | 12 |
| Disability | difficulty walking, dressing, doing errands, concentrating; deaf; blind; arthritis limits activities | 7 |
| Lifestyle | physical activity, smoking, e-cigarettes, alcohol, fruit, vegetables, fried potatoes, BMI | 10 |
| Prevention | flu shot, pneumonia vaccine, HIV test, cholesterol check | 4 |

The most incomplete selected variable is income (21.5 % missing); all the others are under 15 %.

### 2.4 Cleaning and encoding

`encode()` in `brfss.py` applies one rule per variable, taken from the codebook:

| Type | Count | Treatment | Example |
|---|---|---|---|
| Numeric | 6 | special codes → missing; implied decimals removed | `_BMI5` = 2735 → BMI 27.35 |
| Binary | 27 | yes → 1, no → 0, don't know/refused → missing | `DIFFWALK` |
| Ordinal | 8 | ordered ranks (label encoding) | income 1–7; smoking: never 0 → every day 3 |
| Nominal | 8 | one-hot encoding, with "missing" as its own category | employment, marital status, state |

Missing values are imputed **inside each training fold** (median for numeric, binary and ordinal
variables, plus a 0/1 "was missing" indicator; a separate category for nominal ones), and numeric
columns are standardised. Doing this inside the cross-validation pipeline means no information from a
test fold is ever used for training. After encoding, each person is described by **175 columns**.

## 3. Exploratory data analysis

### 3.1 The target is imbalanced

| GENHLTH | Excellent | Very good | Good | Fair | Poor |
|---|---|---|---|---|---|
| People | 77,741 | 149,112 | 137,938 | 54,736 | 18,005 |
| Share | 17.8 % | 34.1 % | 31.5 % | 12.5 % | 4.1 % |

![Target](figures/target.png)

*Figure 3: the five classes and the binary target used in this work.*

The binary positive class (*Fair/Poor*) is **16.6 %** of the people: about **1 in 6**. A model that
always answers "good health" would be right 83.4 % of the time while being useless, which is why
**accuracy is not used**: the main metric is **ROC-AUC** (the probability that a random Fair/Poor person
gets a higher score than a random healthy one), as the guidelines require, completed by the precision-
recall AUC, balanced accuracy and F1, which focus on the minority class.

### 3.2 What the raw data already shows

| Characteristic | Share in Fair/Poor health |
|---|---|
| Difficulty walking: yes / no | 51.7 % / 9.9 % |
| COPD: yes / no | 48.7 % / 13.8 % |
| Depression: yes / no | 30.2 % / 13.2 % |
| Physically inactive / active | 32.7 % / 11.4 % |
| Smoking: never / former / some days / every day | 12.6 / 20.1 / 25.0 / 28.8 % |
| Education: no high school → college graduate | 38.4 % → 9.4 % |
| Income: < $15k → > $200k | 43.0 % → 4.0 % |

![Income and age](figures/income_age.png)

*Figure 4: share of people in Fair/Poor health by household income and age.*

Figure 4 is the clearest pattern in the data: **income matters more than age**. Among people aged 55–64,
54 % of those earning under $15,000 report fair or poor health, against 4 % of those earning over
$200,000. Age matters differently by income: in low-income groups poor health peaks at 55–64 and then
falls (possibly because the sickest die earlier), while in high-income groups it rises steadily with age.

The selected variables are only weakly correlated with each other (Figure 5): of 820 pairs, only one
exceeds 0.5 (binge drinking with drinks per week, 0.53), followed by natural pairs such as age with
pneumonia vaccination (0.48), the disability questions with each other (≈ 0.4) and education with income
(0.43). No variable had to be removed for redundancy.

![Correlation](figures/correlation.png)

*Figure 5: Spearman correlation between the numeric, binary and ordinal variables.*

## 4. Methodology

### 4.1 One validation protocol for every experiment

- **Stratified 10-fold cross-validation** for every experiment, as required, with **the same ten folds
  everywhere** (`StratifiedKFold(10, shuffle=True, random_state=42)`, defined once in `brfss.py`). Each
  model is therefore tested on exactly the same people, which is what makes the Wilcoxon tests of
  Section 6.5 valid.
- Every preprocessing step (imputation, scaling, encoding, feature selection, PCA, resampling) is part of
  a `Pipeline` fitted on the training folds only.
- Results are stored **per fold** (`results/*.csv`), and are reported as mean ± standard deviation.

### 4.2 How much data is needed?

Ten-fold cross-validation of ten models on 437,532 people would take days. A **learning curve** shows
how much data the models really need (Figure 6): models were trained on growing samples and tested on
the same 50,000 unseen people.

![Learning curve](figures/learning_curve.png)

*Figure 6: test ROC-AUC against the number of training rows (log scale). The dashed line marks 50,000.*

| Training rows | 10,000 | 25,000 | **50,000** | 100,000 | 200,000 |
|---|---|---|---|---|---|
| Logistic regression | 0.860 | 0.866 | **0.869** | 0.871 | 0.871 |
| Gradient boosting | 0.861 | 0.869 | **0.872** | 0.875 | 0.876 |

Beyond 50,000 rows, four times more data adds at most 0.004 AUC. The experiments therefore use a
**stratified sample of 50,000 people**, and a separate **hold-out set of 100,000 people** is kept untouched
until the very end (Section 8) to confirm the final model on data never used for any decision.

