# Predicting self-rated health from the BRFSS 2021 survey

Can we tell, from someone's chronic conditions, disabilities, habits and socioeconomic situation,
whether they will describe their own health as *fair or poor*? This project answers that question
with the 2021 Behavioral Risk Factor Surveillance System (BRFSS), the largest continuous health survey
in the world (438,693 interviews), following the
final project guidelines published with the [dataset](https://www.dlsi.ua.es/~juanra/UA/datasets/)
(basic, intermediate and advanced levels).

The report (`report/report.pdf`) explains every step and result; `brfss.py` reproduces all of them.

## Results

Binary target: *Fair/Poor* (16.6 %) against *Excellent/Very good/Good*. Same stratified 10-fold CV
for every model, on a sample of 50,000 people; final check on 100,000 people never used before.

| Model | ROC-AUC |
|---|---|
| Dummy baseline | 0.500 |
| Logistic regression | 0.860 |
| Random forest | 0.864 |
| LightGBM | 0.866 |
| LightGBM, tuned (nested CV) | 0.869 |
| **LightGBM, tuned — hold-out set** | **0.871** |

The most important factors (SHAP) are difficulty walking, physical inactivity, BMI, inability to work,
chronic conditions, income and education. The report also covers class imbalance, PCA, feature
selection, Wilcoxon tests, error analysis, feature engineering and the three-class problem.

## Running

```bash
pip install -r requirements.txt
python brfss.py              # every stage (about 4 hours on a 4-core laptop)
python brfss.py eda models   # or only some stages
```

The script downloads the data and the codebook into `data/` the first time.
Stages: `eda`, `curve`, `basic`, `models`, `imbalance`, `pca`, `selection`, `engineering`, `stats`,
`nested`, `multiclass`, `errors`, `xai`, `holdout`. Each one writes its results to `results/` (CSV) and its
figures to `report/figures/`. All experiments use the same stratified 10-fold split (`random_state=42`).

## Repository

| Path | Contents |
|---|---|
| `brfss.py` | The whole pipeline: cleaning, experiments, figures. |
| `results/` | Raw results of every experiment (one row per fold). |
| `report/` | Report source (`report.md`), figures and `report.pdf`. |
| `data/` | Dataset and codebook (downloaded, not committed). |
