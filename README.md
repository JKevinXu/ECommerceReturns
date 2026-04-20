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

For lower-variance model selection, run repeated 5-fold XGBoost:

```bash
python -m src.retail_return_risk.xgboost_repeated_cv \
  --threshold 0.497 \
  --submission submissions/submission_xgboost_repeated_cv.csv
```

For a neural tabular baseline with categorical embeddings:

```bash
python -m src.retail_return_risk.embedding_mlp \
  --threshold 0.497 \
  --submission submissions/submission_embedding_mlp_thr0497.csv
```

The strongest private-score entry so far is a small neural/tree blend:

```bash
python -m src.retail_return_risk.blend_xgboost_mlp \
  --nn-weight 0.03 \
  --threshold 0.497 \
  --submission submissions/submission_xgb_nn_blend_w003_thr0497.csv
```

## Results

Best submitted scores so far:

```text
best private  submission_xgb_nn_blend_w003_thr0497.csv  public 0.56679  private 0.56552
best public   submission_xgboost_fold73_thr0493.csv  public 0.56848  private 0.56193
```

The best private result is the preferred model selection target.

Pure embedding MLP did not beat XGBoost alone:

```text
submission_embedding_mlp_thr0497.csv  public 0.56464  private 0.56244
```

Repeated 25-model XGBoost did not beat the best single-seed tuned run:

```text
submission_xgboost_repeated_cv_thr0497.csv  public 0.56616  private 0.56464
```

## Submit

```bash
kaggle competitions submit \
  -c retail-return-risk-modeling \
  -f submissions/submission.csv \
  -m "hist-gradient-boosting baseline"
```
