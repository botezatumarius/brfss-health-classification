# Predicting self-rated health from the BRFSS 2021 survey

Machine learning project on the 2021 Behavioral Risk Factor Surveillance System (BRFSS): predict whether
a person describes their own health as *fair or poor* (`GENHLTH`) from their chronic conditions,
disabilities, habits and socioeconomic situation.

## Running

```bash
pip install -r requirements.txt
python brfss.py
```

The script downloads the data and the codebook into `data/` the first time.
