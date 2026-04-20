from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "returned"
ID_COLUMN = "order_id"
SUBMISSION_ID_COLUMN = "ID"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tabular model and write a Kaggle submission."
    )
    parser.add_argument("--train", default="data/raw/train.csv", help="Training CSV path.")
    parser.add_argument("--test", default="data/raw/test.csv", help="Test CSV path.")
    parser.add_argument(
        "--sample",
        default="data/raw/sample_submission.csv",
        help="Sample submission CSV path.",
    )
    parser.add_argument(
        "--submission",
        default="submissions/submission.csv",
        help="Output submission CSV path.",
    )
    parser.add_argument(
        "--model",
        default="models/hist_gradient_boosting.joblib",
        help="Output model artifact path.",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=["accuracy", "f1"],
        default="accuracy",
        help="Validation metric used to choose the probability threshold.",
    )
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_pipeline(x_train: pd.DataFrame, seed: int) -> Pipeline:
    categorical_columns = x_train.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_columns = [column for column in x_train.columns if column not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_columns),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
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

    model = HistGradientBoostingClassifier(
        learning_rate=0.025,
        max_iter=800,
        max_leaf_nodes=63,
        l2_regularization=0.02,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def choose_threshold(
    y_true: pd.Series, probabilities: np.ndarray, metric_name: str
) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (probabilities >= threshold).astype(int)
        if metric_name == "accuracy":
            score = accuracy_score(y_true, predictions)
        else:
            score = f1_score(y_true, predictions)

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, float(best_score)


def build_submission(
    sample_submission: pd.DataFrame,
    test: pd.DataFrame,
    test_probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    if SUBMISSION_ID_COLUMN not in sample_submission.columns:
        raise ValueError(f"Expected sample submission to contain {SUBMISSION_ID_COLUMN!r}.")
    if TARGET_COLUMN not in sample_submission.columns:
        raise ValueError(f"Expected sample submission to contain {TARGET_COLUMN!r}.")
    if ID_COLUMN not in test.columns:
        raise ValueError(f"Expected test data to contain {ID_COLUMN!r}.")

    submission = pd.DataFrame(
        {
            SUBMISSION_ID_COLUMN: test[ID_COLUMN],
            TARGET_COLUMN: (test_probabilities >= threshold).astype(int),
        }
    )

    sample_ids = sample_submission[SUBMISSION_ID_COLUMN].astype(str).tolist()
    submission_ids = submission[SUBMISSION_ID_COLUMN].astype(str).tolist()
    if sample_ids != submission_ids:
        raise ValueError("Test order_id values do not match sample submission ID order.")

    return submission


def main() -> None:
    args = parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    sample_submission = pd.read_csv(args.sample)

    feature_columns = [
        column for column in train.columns if column not in {TARGET_COLUMN, ID_COLUMN}
    ]
    x = train[feature_columns]
    y = train[TARGET_COLUMN]

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=args.valid_size,
        random_state=args.seed,
        stratify=y,
    )

    pipeline = make_pipeline(x_train, args.seed)
    pipeline.fit(x_train, y_train)

    valid_probabilities = pipeline.predict_proba(x_valid)[:, 1]
    threshold, threshold_score = choose_threshold(
        y_valid, valid_probabilities, args.threshold_metric
    )
    valid_predictions = (valid_probabilities >= threshold).astype(int)

    metrics = {
        "validation_accuracy": float(accuracy_score(y_valid, valid_predictions)),
        "validation_f1": float(f1_score(y_valid, valid_predictions)),
        "validation_roc_auc": float(roc_auc_score(y_valid, valid_probabilities)),
        "validation_log_loss": float(log_loss(y_valid, valid_probabilities)),
        "threshold_metric": args.threshold_metric,
        "threshold_metric_score": threshold_score,
        "threshold": threshold,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features": feature_columns,
    }

    final_pipeline = make_pipeline(x, args.seed)
    final_pipeline.fit(x, y)

    test_probabilities = final_pipeline.predict_proba(test[feature_columns])[:, 1]
    submission = build_submission(
        sample_submission=sample_submission,
        test=test,
        test_probabilities=test_probabilities,
        threshold=threshold,
    )

    submission_path = Path(args.submission)
    model_path = Path(args.model)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    submission.to_csv(submission_path, index=False)
    joblib.dump(
        {
            "pipeline": final_pipeline,
            "threshold": threshold,
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
