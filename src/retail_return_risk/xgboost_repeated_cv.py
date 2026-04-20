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


DEFAULT_SEEDS = "42,73,101,137,211"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train repeated K-fold XGBoost models, average probabilities, and write "
            "thresholded Kaggle submissions."
        )
    )
    parser.add_argument("--train", default="data/raw/train.csv")
    parser.add_argument("--test", default="data/raw/test.csv")
    parser.add_argument("--sample", default="data/raw/sample_submission.csv")
    parser.add_argument(
        "--submission",
        default="submissions/submission_xgboost_repeated_cv.csv",
        help="Main submission CSV path.",
    )
    parser.add_argument(
        "--artifact",
        default="models/xgboost_repeated_cv.joblib",
        help="Output artifact with metrics and aggregated probabilities.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--seeds",
        default=DEFAULT_SEEDS,
        help="Comma-separated StratifiedKFold seeds.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold override for the main submission.",
    )
    parser.add_argument(
        "--extra-thresholds",
        default="0.493,0.495,0.497",
        help="Comma-separated extra thresholds to write beside the main submission.",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=["accuracy", "f1"],
        default="accuracy",
    )
    parser.add_argument(
        "--keep-models",
        action="store_true",
        help="Persist trained fold models in the artifact. This makes the file larger.",
    )
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    if not raw.strip():
        return []
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def parse_int_list(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def make_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = x.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_columns = [column for column in x.columns if column not in categorical_columns]

    return ColumnTransformer(
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


def make_model(seed: int) -> XGBClassifier:
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


def threshold_suffix(threshold: float) -> str:
    return str(round(threshold, 6)).replace(".", "")


def write_threshold_submission(
    sample_submission: pd.DataFrame,
    test: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    output_path: Path,
) -> dict[str, float | int | str]:
    submission = build_submission(
        sample_submission=sample_submission,
        test=test,
        test_probabilities=probabilities,
        threshold=threshold,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    positives = int(submission[TARGET_COLUMN].sum())
    return {
        "path": str(output_path),
        "threshold": float(threshold),
        "positive_predictions": positives,
        "positive_rate": float(positives / len(submission)),
    }


def main() -> None:
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("At least one seed is required.")

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    sample_submission = pd.read_csv(args.sample)

    feature_columns = [
        column for column in train.columns if column not in {TARGET_COLUMN, ID_COLUMN}
    ]
    x = train[feature_columns]
    y = train[TARGET_COLUMN]
    test_x = test[feature_columns]

    oof_sum = np.zeros(len(train), dtype=float)
    test_probabilities = np.zeros(len(test), dtype=float)
    fold_metrics = []
    models = []

    total_folds = len(seeds) * args.folds
    for seed_index, seed in enumerate(seeds, start=1):
        folds = StratifiedKFold(
            n_splits=args.folds,
            shuffle=True,
            random_state=seed,
        )
        seed_oof = np.zeros(len(train), dtype=float)

        for fold, (train_index, valid_index) in enumerate(folds.split(x, y), start=1):
            x_train = x.iloc[train_index]
            x_valid = x.iloc[valid_index]
            y_train = y.iloc[train_index]
            y_valid = y.iloc[valid_index]

            model_seed = seed * 1000 + fold
            pipeline = Pipeline(
                steps=[
                    ("preprocess", make_preprocessor(x_train)),
                    ("model", make_model(model_seed)),
                ]
            )
            completed = (seed_index - 1) * args.folds + fold
            print(
                f"Fitting seed {seed} fold {fold}/{args.folds} "
                f"({completed}/{total_folds})",
                flush=True,
            )
            pipeline.fit(x_train, y_train)

            valid_probabilities = pipeline.predict_proba(x_valid)[:, 1]
            seed_oof[valid_index] = valid_probabilities
            test_probabilities += pipeline.predict_proba(test_x)[:, 1] / total_folds

            fold_threshold, fold_score = choose_threshold(
                y_valid, valid_probabilities, args.threshold_metric
            )
            fold_predictions = (valid_probabilities >= fold_threshold).astype(int)
            fold_metrics.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "threshold": fold_threshold,
                    "threshold_score": float(fold_score),
                    "accuracy": float(accuracy_score(y_valid, fold_predictions)),
                    "f1": float(f1_score(y_valid, fold_predictions)),
                    "roc_auc": float(roc_auc_score(y_valid, valid_probabilities)),
                    "log_loss": float(log_loss(y_valid, valid_probabilities)),
                }
            )
            if args.keep_models:
                models.append(pipeline)

        seed_threshold, seed_score = choose_threshold(
            y, seed_oof, args.threshold_metric
        )
        print(
            f"Seed {seed} OOF {args.threshold_metric}: "
            f"{seed_score:.6f} at threshold {seed_threshold:.6f}",
            flush=True,
        )
        oof_sum += seed_oof / len(seeds)

    oof_threshold, threshold_score = choose_threshold(
        y, oof_sum, args.threshold_metric
    )
    oof_predictions = (oof_sum >= oof_threshold).astype(int)
    submission_threshold = args.threshold if args.threshold is not None else oof_threshold

    metrics = {
        "oof_accuracy": float(accuracy_score(y, oof_predictions)),
        "oof_f1": float(f1_score(y, oof_predictions)),
        "oof_roc_auc": float(roc_auc_score(y, oof_sum)),
        "oof_log_loss": float(log_loss(y, oof_sum)),
        "threshold_metric": args.threshold_metric,
        "threshold_metric_score": float(threshold_score),
        "oof_threshold": float(oof_threshold),
        "submission_threshold": float(submission_threshold),
        "folds": args.folds,
        "seeds": seeds,
        "total_models": total_folds,
        "fold_metrics": fold_metrics,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features": feature_columns,
    }

    submission_outputs = []
    main_path = Path(args.submission)
    submission_outputs.append(
        write_threshold_submission(
            sample_submission,
            test,
            test_probabilities,
            submission_threshold,
            main_path,
        )
    )

    for threshold in parse_float_list(args.extra_thresholds):
        if np.isclose(threshold, submission_threshold):
            continue
        output_path = main_path.with_name(
            f"{main_path.stem}_thr{threshold_suffix(threshold)}{main_path.suffix}"
        )
        submission_outputs.append(
            write_threshold_submission(
                sample_submission,
                test,
                test_probabilities,
                threshold,
                output_path,
            )
        )

    artifact_path = Path(args.artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "metrics": metrics,
        "feature_columns": feature_columns,
        "oof_probabilities": oof_sum,
        "test_probabilities": test_probabilities,
        "submission_outputs": submission_outputs,
    }
    if args.keep_models:
        artifact["models"] = models
    joblib.dump(artifact, artifact_path)

    print(json.dumps(metrics, indent=2))
    print("Wrote submissions:")
    for output in submission_outputs:
        print(json.dumps(output), flush=True)
    print(f"Wrote artifact: {artifact_path}")


if __name__ == "__main__":
    main()

