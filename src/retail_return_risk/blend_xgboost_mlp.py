from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.retail_return_risk.train import build_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blend tuned XGBoost probabilities with embedding-MLP probabilities."
    )
    parser.add_argument("--test", default="data/raw/test.csv")
    parser.add_argument("--sample", default="data/raw/sample_submission.csv")
    parser.add_argument("--xgboost-artifact", default="models/xgboost_tuned.joblib")
    parser.add_argument("--mlp-artifact", default="models/embedding_mlp.joblib")
    parser.add_argument(
        "--submission",
        default="submissions/submission_xgb_nn_blend.csv",
    )
    parser.add_argument("--nn-weight", type=float, default=0.03)
    parser.add_argument("--threshold", type=float, default=0.497)
    parser.add_argument("--extra-thresholds", default="")
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    if not raw.strip():
        return []
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def threshold_suffix(value: float) -> str:
    return str(round(value, 6)).replace(".", "")


def load_xgboost_probabilities(path: str, test: pd.DataFrame) -> np.ndarray:
    artifact = joblib.load(path)
    feature_columns = artifact["feature_columns"]
    probabilities = [
        model.predict_proba(test[feature_columns])[:, 1]
        for model in artifact["models"]
    ]
    return np.mean(probabilities, axis=0)


def write_submission(
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
    positives = int(submission["returned"].sum())
    return {
        "path": str(output_path),
        "threshold": float(threshold),
        "positive_predictions": positives,
        "positive_rate": float(positives / len(submission)),
    }


def main() -> None:
    args = parse_args()

    test = pd.read_csv(args.test)
    sample_submission = pd.read_csv(args.sample)
    xgboost_probabilities = load_xgboost_probabilities(args.xgboost_artifact, test)
    mlp_artifact = joblib.load(args.mlp_artifact)
    mlp_probabilities = mlp_artifact["test_probabilities"]

    blend_probabilities = (
        (1 - args.nn_weight) * xgboost_probabilities
        + args.nn_weight * mlp_probabilities
    )
    diagnostics = {
        "nn_weight": args.nn_weight,
        "xgboost_mean_probability": float(np.mean(xgboost_probabilities)),
        "mlp_mean_probability": float(np.mean(mlp_probabilities)),
        "blend_mean_probability": float(np.mean(blend_probabilities)),
        "xgboost_mlp_correlation": float(
            np.corrcoef(xgboost_probabilities, mlp_probabilities)[0, 1]
        ),
    }
    print(json.dumps(diagnostics, indent=2))

    outputs = []
    submission_path = Path(args.submission)
    outputs.append(
        write_submission(
            sample_submission,
            test,
            blend_probabilities,
            args.threshold,
            submission_path,
        )
    )
    for threshold in parse_float_list(args.extra_thresholds):
        if np.isclose(threshold, args.threshold):
            continue
        output_path = submission_path.with_name(
            f"{submission_path.stem}_thr{threshold_suffix(threshold)}"
            f"{submission_path.suffix}"
        )
        outputs.append(
            write_submission(
                sample_submission,
                test,
                blend_probabilities,
                threshold,
                output_path,
            )
        )

    print("Wrote submissions:")
    for output in outputs:
        print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()

