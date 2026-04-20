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

The strongest submitted variant so far used a slightly lower threshold:

```bash
python -m src.retail_return_risk.xgboost_tuned \
  --threshold 0.495 \
  --submission submissions/submission_xgboost_tuned_thr0495.csv
```

## Results

Best submitted scores so far:

```text
best private  submission_xgboost_tuned_thr0495.csv  public 0.56807  private 0.56499
best public   submission_xgboost_tuned_thr049.csv   public 0.56835  private 0.56411
```

## Submit

```bash
kaggle competitions submit \
  -c retail-return-risk-modeling \
  -f submissions/submission.csv \
  -m "hist-gradient-boosting baseline"
```
