# ECommerceReturns

Starter workflow for Kaggle's "E-Commerce Returns: Who Will Send It Back?"
competition:

https://www.kaggle.com/competitions/retail-return-risk-modeling

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data

```bash
mkdir -p data/raw
kaggle competitions download -c retail-return-risk-modeling -p data/raw --force
unzip -o data/raw/retail-return-risk-modeling.zip -d data/raw
```

The raw Kaggle files are intentionally ignored by git.

## Train

```bash
source .venv/bin/activate
python -m src.retail_return_risk.train
```

This writes:

- `submissions/submission.csv`
- `models/hist_gradient_boosting.joblib`

For a stronger local baseline, run the small ensemble:

```bash
python -m src.retail_return_risk.ensemble
```

For the tuned XGBoost model with cross-validated threshold selection:

```bash
python -m src.retail_return_risk.xgboost_tuned
```

The strongest private-score variant so far used a slightly lower threshold:

```bash
python -m src.retail_return_risk.xgboost_tuned \
  --threshold 0.497 \
  --submission submissions/submission_xgboost_tuned_thr0497.csv
```

## Results

Best submitted scores so far:

```text
best private  submission_xgboost_tuned_thr0497.csv   public 0.56707  private 0.56541
best public   submission_xgboost_fold73_thr0493.csv  public 0.56848  private 0.56193
```

The best private result is the preferred model selection target.

## Submit

```bash
kaggle competitions submit \
  -c retail-return-risk-modeling \
  -f submissions/submission.csv \
  -m "hist-gradient-boosting baseline"
```
