from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.retail_return_risk.train import (
    ID_COLUMN,
    SUBMISSION_ID_COLUMN,
    TARGET_COLUMN,
    build_submission,
    choose_threshold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small model ensemble and write a Kaggle submission."
    )
    parser.add_argument("--train", default="data/raw/train.csv")
    parser.add_argument("--test", default="data/raw/test.csv")
    parser.add_argument("--sample", default="data/raw/sample_submission.csv")
    parser.add_argument("--submission", default="submissions/submission_ensemble.csv")
    parser.add_argument("--model", default="models/ensemble.joblib")
    parser.add_argument(
        "--threshold-metric",
        choices=["accuracy", "f1"],
        default="accuracy",
    )
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_preprocessor(
    x_train: pd.DataFrame, scale_numeric: bool = False
) -> ColumnTransformer:
    categorical_columns = x_train.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_columns = [column for column in x_train.columns if column not in categorical_columns]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_columns),
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


def make_models(x_train: pd.DataFrame, seed: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(x_train, scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        C=0.5,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting_a": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(x_train)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.04,
                        max_iter=300,
                        max_leaf_nodes=31,
                        l2_regularization=0.05,
                        random_state=seed,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=30,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting_b": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(x_train)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.025,
                        max_iter=800,
                        max_leaf_nodes=63,
                        l2_regularization=0.02,
                        random_state=seed + 1,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=35,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(x_train)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        max_features="sqrt",
                        min_samples_leaf=20,
                        n_jobs=-1,
                        random_state=seed + 2,
                        class_weight="balanced_subsample",
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(x_train)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_features="sqrt",
                        min_samples_leaf=30,
                        n_jobs=-1,
                        random_state=seed + 3,
                        class_weight="balanced_subsample",
                    ),
                ),
            ]
        ),
    }


def average_probabilities(models: dict[str, Pipeline], x: pd.DataFrame) -> np.ndarray:
    probabilities = [model.predict_proba(x)[:, 1] for model in models.values()]
    return np.mean(probabilities, axis=0)


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

    validation_models = make_models(x_train, args.seed)
    for name, model in validation_models.items():
        print(f"Fitting validation model: {name}", flush=True)
        model.fit(x_train, y_train)

    valid_probabilities = average_probabilities(validation_models, x_valid)
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
        "models": list(validation_models),
    }

    final_models = make_models(x, args.seed)
    for name, model in final_models.items():
        print(f"Fitting final model: {name}", flush=True)
        model.fit(x, y)

    test_probabilities = average_probabilities(final_models, test[feature_columns])
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
            "models": final_models,
            "threshold": threshold,
            "metrics": metrics,
            "feature_columns": feature_columns,
            "submission_id_column": SUBMISSION_ID_COLUMN,
        },
        model_path,
    )

    print(json.dumps(metrics, indent=2))
    print(f"Wrote submission: {submission_path}")
    print(f"Wrote model: {model_path}")


if __name__ == "__main__":
    main()

