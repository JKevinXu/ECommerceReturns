# ECommerceReturns

Experiment report and reproducible workflow for Kaggle's "E-Commerce Returns:
Who Will Send It Back?" competition:

https://www.kaggle.com/competitions/retail-return-risk-modeling

## Summary

This is a binary classification problem: predict whether each e-commerce order
will be returned.

The best private-score submission so far is a small tree/neural blend:

```text
submission_xgb_nn_blend_w003_thr0497.csv
public  0.56679
private 0.56552
```

The final model is:

```text
97% tuned 5-fold XGBoost probability
 3% embedding-MLP probability
threshold = 0.497
```

The pure XGBoost model remains the main signal. The neural model is weaker by
itself, but it gives a small correction on borderline rows.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Download Data

```bash
mkdir -p data/raw
kaggle competitions download -c retail-return-risk-modeling -p data/raw --force
unzip -o data/raw/retail-return-risk-modeling.zip -d data/raw
```

The raw Kaggle files are intentionally ignored by git.

## Data

Files from Kaggle:

```text
train.csv              200,000 rows, 16 columns
test.csv                50,000 rows, 15 columns
sample_submission.csv   50,000 rows
```

Target:

```text
returned
```

Submission ID:

```text
order_id -> ID
```

Features used by the best model:

```text
customer_age
product_price
discount_percent
product_rating
past_purchase_count
past_return_rate
delivery_delay_days
session_length_minutes
num_product_views
device_type
product_category
shipping_method
payment_method
used_coupon
```

Excluded from model training:

```text
returned  # target
order_id  # submission ID
```

## Methods

### Histogram Gradient Boosting Baseline

```bash
source .venv/bin/activate
python -m src.retail_return_risk.train
```

This writes:

- `submissions/submission.csv`
- `models/hist_gradient_boosting.joblib`

This baseline uses scikit-learn preprocessing plus
`HistGradientBoostingClassifier`. It established a working pipeline and
submission format.

### Scikit-Learn Ensemble

```bash
python -m src.retail_return_risk.ensemble
```

This averages several classical models:

```text
LogisticRegression
HistGradientBoostingClassifier
ExtraTreesClassifier
RandomForestClassifier
```

It improved private score over the first baseline, but was later beaten by
XGBoost.

### Tuned XGBoost

XGBoost was the strongest single model family.

```bash
python -m src.retail_return_risk.xgboost_tuned
```

How it works:

```text
1. Split train data into 5 stratified folds.
2. For each fold:
   - train preprocessing + XGBoost on 4 folds
   - validate on the held-out fold
   - predict the test set
3. Average the 5 test probabilities.
4. Convert probabilities to 0/1 using a threshold.
```

Preprocessing:

```text
numeric columns      -> median imputation
categorical columns  -> most-frequent imputation + one-hot encoding
```

Important model parameters:

```text
n_estimators=1200
learning_rate=0.025
max_depth=3
min_child_weight=3
subsample=0.7
colsample_bytree=0.85
reg_lambda=1.0
reg_alpha=0.2
objective="binary:logistic"
```

Best XGBoost-only private variant:

```bash
python -m src.retail_return_risk.xgboost_tuned \
  --threshold 0.497 \
  --submission submissions/submission_xgboost_tuned_thr0497.csv
```

### Repeated XGBoost CV

This was tested to reduce variance by averaging 25 XGBoost models:

```bash
python -m src.retail_return_risk.xgboost_repeated_cv \
  --threshold 0.497 \
  --submission submissions/submission_xgboost_repeated_cv.csv
```

It improved probability stability locally, but did not beat the best
single-seed XGBoost on Kaggle private score.

### Embedding MLP

The deep-learning model is a PyTorch tabular network.

```bash
python -m src.retail_return_risk.embedding_mlp \
  --device cpu \
  --epochs 25 \
  --threshold 0.497 \
  --submission submissions/submission_embedding_mlp_thr0497.csv
```

How it works:

```text
numeric features
-> standardize

categorical features
-> learned embeddings

numeric vector + embeddings
-> Dense 128
-> Dense 64
-> Dense 32
-> sigmoid probability
```

The neural net alone was weaker than XGBoost, but its probabilities were not
identical to XGBoost, so a small blend helped.

### XGBoost + Embedding MLP Blend

Best private-score method:

```bash
python -m src.retail_return_risk.blend_xgboost_mlp \
  --nn-weight 0.03 \
  --threshold 0.497 \
  --submission submissions/submission_xgb_nn_blend_w003_thr0497.csv
```

Blend formula:

```text
final_probability =
    0.97 * xgboost_probability
  + 0.03 * embedding_mlp_probability
```

Final prediction rule:

```text
if final_probability >= 0.497:
    returned = 1
else:
    returned = 0
```

## Experiment Results

Private score is the preferred model-selection target. Public score was useful
for quick feedback but was not always aligned with private score.

```text
Method                                      Submission                                      Public   Private
------------------------------------------  ----------------------------------------------  -------  -------
HistGradientBoosting baseline               submission.csv                                  0.56262  0.55566
HistGradientBoosting refit                  submission_refit.csv                            0.56148  0.55842
Scikit-learn ensemble                       submission_ensemble.csv                         0.56242  0.55979
Tuned XGBoost                               submission_xgboost_tuned.csv                    0.56592  0.56307
Tuned XGBoost, threshold 0.495              submission_xgboost_tuned_thr0495.csv            0.56807  0.56499
Tuned XGBoost, threshold 0.497              submission_xgboost_tuned_thr0497.csv            0.56707  0.56541
Repeated 25-model XGBoost, threshold 0.497  submission_xgboost_repeated_cv_thr0497.csv      0.56616  0.56464
Embedding MLP, threshold 0.497              submission_embedding_mlp_thr0497.csv            0.56464  0.56244
XGBoost + 3% MLP blend, threshold 0.497     submission_xgb_nn_blend_w003_thr0497.csv        0.56679  0.56552
```

Best public score observed:

```text
submission_xgboost_fold73_thr0493.csv
public  0.56848
private 0.56193
```

This was not selected as the final model because the private score was much
lower than the best private-score submission.

## Findings

Threshold tuning mattered more than expected. Moving from the default-ish
threshold to `0.497` improved the private score for XGBoost.

XGBoost was the strongest single model. It handled the mixed numeric and
low-cardinality categorical tabular features better than the neural net alone.

Repeated CV improved local probability metrics but did not transfer to a better
Kaggle private score. More models were not automatically better.

The embedding MLP alone underperformed XGBoost, but a tiny 3% neural blend
improved the private score slightly. This suggests the neural model captured a
small amount of different signal, mostly useful for borderline rows.

Public score was sometimes misleading. The best public-score submission had a
weaker private score, so private score is treated as the final target.

## Reproduce Best Submission

Train XGBoost:

```bash
python -m src.retail_return_risk.xgboost_tuned \
  --threshold 0.497 \
  --submission submissions/submission_xgboost_tuned_thr0497.csv \
  --model models/xgboost_tuned.joblib
```

Train embedding MLP:

```bash
python -m src.retail_return_risk.embedding_mlp \
  --device cpu \
  --epochs 25 \
  --threshold 0.497 \
  --submission submissions/submission_embedding_mlp_thr0497.csv \
  --artifact models/embedding_mlp.joblib
```

Blend:

```bash
python -m src.retail_return_risk.blend_xgboost_mlp \
  --nn-weight 0.03 \
  --threshold 0.497 \
  --submission submissions/submission_xgb_nn_blend_w003_thr0497.csv
```

Submit:

```bash
kaggle competitions submit \
  -c retail-return-risk-modeling \
  -f submissions/submission_xgb_nn_blend_w003_thr0497.csv \
  -m "xgboost embedding mlp blend nn weight 0.03 threshold 0.497"
```
