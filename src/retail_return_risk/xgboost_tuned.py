from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.retail_return_risk.train import (
    ID_COLUMN,
    TARGET_COLUMN,
    build_submission,
    choose_threshold,
)


def parse_args() -> argparse.Namespace:
    # Keep file paths and training choices configurable from the command line.
    parser = argparse.ArgumentParser(
        description="Train a tuned XGBoost model and write a Kaggle submission."
    )
    parser.add_argument("--train", default="data/raw/train.csv")
    parser.add_argument("--test", default="data/raw/test.csv")
    parser.add_argument("--sample", default="data/raw/sample_submission.csv")
    parser.add_argument("--submission", default="submissions/submission_xgboost_tuned.csv")
    parser.add_argument("--model", default="models/xgboost_tuned.joblib")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold override for the written submission.",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=["accuracy", "f1"],
        default="accuracy",
    )
    return parser.parse_args()


def make_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    # XGBoost needs numeric arrays, so split raw columns by data type first.
    categorical_columns = x.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_columns = [column for column in x.columns if column not in categorical_columns]

    return ColumnTransformer(
        transformers=[
            # Numeric features stay numeric; missing values are replaced by the fold median.
            ("num", SimpleImputer(strategy="median"), numeric_columns),
            (
                "cat",
                Pipeline(
                    steps=[
                        # Categorical missing values become the most common category in the fold.
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            # One-hot encoding turns strings like "mobile" into 0/1 columns.
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_model(seed: int) -> XGBClassifier:
    # Shallow, regularized trees work well for this noisy tabular problem.
    return XGBClassifier(
        n_estimators=1200,
        learning_rate=0.025,
        max_depth=3,
        min_child_weight=3,
        subsample=0.7,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.2,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )


def main() -> None:
    args = parse_args()

    # Kaggle provides train labels, unlabeled test rows, and a required ID order.
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    sample_submission = pd.read_csv(args.sample)

    # Use all predictive columns, but never train on the target or submission ID.
    feature_columns = [
        column for column in train.columns if column not in {TARGET_COLUMN, ID_COLUMN}
    ]
    x = train[feature_columns]
    y = train[TARGET_COLUMN]
    test_x = test[feature_columns]

    # Stratified folds preserve the returned/not-returned balance in every split.
    folds = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    # OOF predictions score rows with models that did not train on those rows.
    oof_probabilities = np.zeros(len(train), dtype=float)
    # Test predictions are averaged across fold models for a more stable submission.
    test_probabilities = np.zeros(len(test), dtype=float)
    models = []
    fold_metrics = []

    for fold, (train_index, valid_index) in enumerate(folds.split(x, y), start=1):
        # Each loop trains on 4 folds and validates on the held-out fold.
        x_train = x.iloc[train_index]
        x_valid = x.iloc[valid_index]
        y_train = y.iloc[train_index]
        y_valid = y.iloc[valid_index]

        # The pipeline guarantees validation/test data receive the same preprocessing.
        pipeline = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(x_train)),
                ("model", make_model(args.seed + fold)),
            ]
        )
        print(f"Fitting fold {fold}/{args.folds}", flush=True)
        pipeline.fit(x_train, y_train)

        # Column 1 is P(returned = 1); column 0 would be P(returned = 0).
        valid_probabilities = pipeline.predict_proba(x_valid)[:, 1]
        oof_probabilities[valid_index] = valid_probabilities
        test_probabilities += pipeline.predict_proba(test_x)[:, 1] / args.folds
        models.append(pipeline)

        # Pick the best cutoff for this fold and store fold-level diagnostics.
        fold_threshold, fold_accuracy = choose_threshold(
            y_valid, valid_probabilities, args.threshold_metric
        )
        fold_predictions = (valid_probabilities >= fold_threshold).astype(int)
        fold_metrics.append(
            {
                "fold": fold,
                "accuracy": float(fold_accuracy),
                "threshold": fold_threshold,
                "f1": float(f1_score(y_valid, fold_predictions)),
                "roc_auc": float(roc_auc_score(y_valid, valid_probabilities)),
                "log_loss": float(log_loss(y_valid, valid_probabilities)),
            }
        )

    # Choose an OOF threshold for reporting; CLI --threshold can override submission cutoff.
    oof_threshold, threshold_score = choose_threshold(
        y, oof_probabilities, args.threshold_metric
    )
    submission_threshold = args.threshold if args.threshold is not None else oof_threshold
    oof_predictions = (oof_probabilities >= oof_threshold).astype(int)
    metrics = {
        "oof_accuracy": float(accuracy_score(y, oof_predictions)),
        "oof_f1": float(f1_score(y, oof_predictions)),
        "oof_roc_auc": float(roc_auc_score(y, oof_probabilities)),
        "oof_log_loss": float(log_loss(y, oof_probabilities)),
        "threshold_metric": args.threshold_metric,
        "threshold_metric_score": threshold_score,
        "oof_threshold": oof_threshold,
        "submission_threshold": submission_threshold,
        "folds": args.folds,
        "fold_metrics": fold_metrics,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features": feature_columns,
    }

    # Convert averaged test probabilities into Kaggle's required ID,returned CSV.
    submission = build_submission(
        sample_submission=sample_submission,
        test=test,
        test_probabilities=test_probabilities,
        threshold=submission_threshold,
    )

    submission_path = Path(args.submission)
    model_path = Path(args.model)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    submission.to_csv(submission_path, index=False)
    # Store trained fold pipelines so later scripts can reuse their probabilities.
    joblib.dump(
        {
            "models": models,
            "oof_threshold": oof_threshold,
            "submission_threshold": submission_threshold,
            "metrics": metrics,
            "feature_columns": feature_columns,
        },
        model_path,
    )

    print(json.dumps(metrics, indent=2))
    print(f"Wrote submission: {submission_path}")
    print(f"Wrote model: {model_path}")


if __name__ == "__main__":
    main()
