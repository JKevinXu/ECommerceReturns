from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.retail_return_risk.train import (
    ID_COLUMN,
    TARGET_COLUMN,
    build_submission,
    choose_threshold,
)


@dataclass
class FoldPreprocessor:
    numeric_columns: list[str]
    categorical_columns: list[str]
    medians: pd.Series
    means: pd.Series
    stds: pd.Series
    category_maps: dict[str, dict[str, int]]

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        numeric = frame[self.numeric_columns].copy()
        numeric = numeric.fillna(self.medians)
        numeric = (numeric - self.means) / self.stds
        numeric = numeric.to_numpy(dtype=np.float32)

        categorical_parts = []
        for column in self.categorical_columns:
            values = frame[column].astype("string").fillna("__MISSING__")
            codes = values.map(self.category_maps[column]).fillna(0).to_numpy(dtype=np.int64)
            categorical_parts.append(codes)

        categorical = np.stack(categorical_parts, axis=1).astype(np.int64)
        return numeric, categorical

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.category_maps[column]) + 1 for column in self.categorical_columns]


class TabularDataset(Dataset):
    def __init__(
        self,
        numeric: np.ndarray,
        categorical: np.ndarray,
        target: np.ndarray | None = None,
    ) -> None:
        self.numeric = torch.from_numpy(numeric)
        self.categorical = torch.from_numpy(categorical)
        self.target = None if target is None else torch.from_numpy(target.astype(np.float32))

    def __len__(self) -> int:
        return len(self.numeric)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.target is None:
            return self.numeric[index], self.categorical[index], torch.tensor(0.0)
        return self.numeric[index], self.categorical[index], self.target[index]


class TabularEmbeddingMLP(nn.Module):
    def __init__(
        self,
        numeric_dim: int,
        cardinalities: list[int],
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        embedding_dims = [min(16, max(2, (cardinality + 1) // 2)) for cardinality in cardinalities]
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, embedding_dim)
                for cardinality, embedding_dim in zip(cardinalities, embedding_dims)
            ]
        )

        input_dim = numeric_dim + sum(embedding_dims)
        layers: list[nn.Module] = []
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        embeddings = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        features = torch.cat([numeric, *embeddings], dim=1)
        return self.network(features).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an embedding MLP tabular model and write Kaggle submissions."
    )
    parser.add_argument("--train", default="data/raw/train.csv")
    parser.add_argument("--test", default="data/raw/test.csv")
    parser.add_argument("--sample", default="data/raw/sample_submission.csv")
    parser.add_argument(
        "--submission",
        default="submissions/submission_embedding_mlp.csv",
    )
    parser.add_argument("--artifact", default="models/embedding_mlp.joblib")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--hidden-dims", default="128,64,32")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument(
        "--keep-models",
        action="store_true",
        help="Persist fold model state dictionaries in the artifact.",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--extra-thresholds", default="0.493,0.495,0.497,0.5")
    parser.add_argument(
        "--xgboost-artifact",
        default="",
        help="Optional XGBoost artifact used to write neural/tree blend submissions.",
    )
    parser.add_argument(
        "--blend-weights",
        default="",
        help="Neural-net weights for test-time blending with XGBoost probabilities.",
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


def threshold_suffix(threshold: float) -> str:
    return str(round(threshold, 6)).replace(".", "")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        return torch.device("mps")
    if name == "cuda":
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def fit_preprocessor(x_train: pd.DataFrame) -> FoldPreprocessor:
    categorical_columns = x_train.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_columns = [column for column in x_train.columns if column not in categorical_columns]

    medians = x_train[numeric_columns].median()
    filled = x_train[numeric_columns].fillna(medians)
    means = filled.mean()
    stds = filled.std().replace(0, 1.0)

    category_maps = {}
    for column in categorical_columns:
        values = x_train[column].astype("string").fillna("__MISSING__")
        categories = sorted(values.unique().tolist())
        category_maps[column] = {category: index + 1 for index, category in enumerate(categories)}

    return FoldPreprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        medians=medians,
        means=means,
        stds=stds,
        category_maps=category_maps,
    )


def make_loader(
    numeric: np.ndarray,
    categorical: np.ndarray,
    target: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TabularDataset(numeric, categorical, target),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def predict_probabilities(
    model: TabularEmbeddingMLP,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for numeric, categorical, _ in loader:
            numeric = numeric.to(device)
            categorical = categorical.to(device)
            logits = model(numeric, categorical)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            predictions.append(probabilities)
    return np.concatenate(predictions)


def train_fold(
    model: TabularEmbeddingMLP,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    y_valid: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TabularEmbeddingMLP, dict[str, float | int]]:
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for numeric, categorical, target in train_loader:
            numeric = numeric.to(device)
            categorical = categorical.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(numeric, categorical)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            rows = len(target)
            total_loss += float(loss.detach().cpu()) * rows
            total_rows += rows

        valid_probabilities = predict_probabilities(model, valid_loader, device)
        valid_loss = log_loss(y_valid, valid_probabilities, labels=[0, 1])
        scheduler.step(valid_loss)

        if valid_loss < best_loss - 1e-5:
            best_loss = valid_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1

        print(
            f"  epoch {epoch:02d} train_loss={total_loss / total_rows:.6f} "
            f"valid_log_loss={valid_loss:.6f}",
            flush=True,
        )
        if stale_epochs >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_log_loss": float(best_loss)}


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
    positives = int(submission[TARGET_COLUMN].sum())
    return {
        "path": str(output_path),
        "threshold": float(threshold),
        "positive_predictions": positives,
        "positive_rate": float(positives / len(submission)),
    }


def load_xgboost_test_probabilities(
    artifact_path: str,
    test_x: pd.DataFrame,
) -> np.ndarray | None:
    if not artifact_path.strip():
        return None
    path = Path(artifact_path)
    if not path.exists():
        return None
    artifact = joblib.load(path)
    if "models" not in artifact:
        return None
    probabilities = [
        model.predict_proba(test_x[artifact["feature_columns"]])[:, 1]
        for model in artifact["models"]
    ]
    return np.mean(probabilities, axis=0)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    hidden_dims = parse_int_list(args.hidden_dims)

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    sample_submission = pd.read_csv(args.sample)

    feature_columns = [
        column for column in train.columns if column not in {TARGET_COLUMN, ID_COLUMN}
    ]
    x = train[feature_columns]
    y = train[TARGET_COLUMN].to_numpy(dtype=np.float32)
    test_x = test[feature_columns]
    device = get_device(args.device)
    print(f"Using device: {device}", flush=True)

    folds = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    oof_probabilities = np.zeros(len(train), dtype=float)
    test_probabilities = np.zeros(len(test), dtype=float)
    fold_metrics = []
    model_states = []

    for fold, (train_index, valid_index) in enumerate(folds.split(x, y), start=1):
        print(f"Fitting fold {fold}/{args.folds}", flush=True)
        set_seed(args.seed + fold)

        x_train = x.iloc[train_index]
        x_valid = x.iloc[valid_index]
        y_train = y[train_index]
        y_valid = y[valid_index]

        preprocessor = fit_preprocessor(x_train)
        train_numeric, train_categorical = preprocessor.transform(x_train)
        valid_numeric, valid_categorical = preprocessor.transform(x_valid)
        test_numeric, test_categorical = preprocessor.transform(test_x)

        train_loader = make_loader(
            train_numeric,
            train_categorical,
            y_train,
            args.batch_size,
            shuffle=True,
        )
        valid_loader = make_loader(
            valid_numeric,
            valid_categorical,
            y_valid,
            args.batch_size,
            shuffle=False,
        )
        test_loader = make_loader(
            test_numeric,
            test_categorical,
            None,
            args.batch_size,
            shuffle=False,
        )

        model = TabularEmbeddingMLP(
            numeric_dim=train_numeric.shape[1],
            cardinalities=preprocessor.cardinalities,
            hidden_dims=hidden_dims,
            dropout=args.dropout,
        )
        model, training_metrics = train_fold(
            model,
            train_loader,
            valid_loader,
            y_valid,
            args,
            device,
        )

        valid_probabilities = predict_probabilities(model, valid_loader, device)
        test_probabilities += predict_probabilities(model, test_loader, device) / args.folds
        oof_probabilities[valid_index] = valid_probabilities

        fold_threshold, fold_score = choose_threshold(
            pd.Series(y_valid.astype(int)),
            valid_probabilities,
            "accuracy",
        )
        fold_predictions = (valid_probabilities >= fold_threshold).astype(int)
        fold_metrics.append(
            {
                "fold": fold,
                "threshold": fold_threshold,
                "threshold_score": float(fold_score),
                "accuracy": float(accuracy_score(y_valid, fold_predictions)),
                "f1": float(f1_score(y_valid, fold_predictions)),
                "roc_auc": float(roc_auc_score(y_valid, valid_probabilities)),
                "log_loss": float(log_loss(y_valid, valid_probabilities)),
                **training_metrics,
            }
        )
        if args.keep_models:
            model_states.append(
                {
                    "preprocessor": preprocessor,
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "numeric_dim": train_numeric.shape[1],
                    "cardinalities": preprocessor.cardinalities,
                }
            )

    oof_threshold, threshold_score = choose_threshold(
        pd.Series(y.astype(int)),
        oof_probabilities,
        "accuracy",
    )
    submission_threshold = args.threshold if args.threshold is not None else oof_threshold
    oof_predictions = (oof_probabilities >= oof_threshold).astype(int)
    metrics = {
        "oof_accuracy": float(accuracy_score(y, oof_predictions)),
        "oof_f1": float(f1_score(y, oof_predictions)),
        "oof_roc_auc": float(roc_auc_score(y, oof_probabilities)),
        "oof_log_loss": float(log_loss(y, oof_probabilities)),
        "threshold_metric": "accuracy",
        "threshold_metric_score": float(threshold_score),
        "oof_threshold": float(oof_threshold),
        "submission_threshold": float(submission_threshold),
        "folds": args.folds,
        "epochs": args.epochs,
        "patience": args.patience,
        "hidden_dims": hidden_dims,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "fold_metrics": fold_metrics,
        "features": feature_columns,
    }

    submission_path = Path(args.submission)
    submission_outputs = [
        write_submission(
            sample_submission,
            test,
            test_probabilities,
            submission_threshold,
            submission_path,
        )
    ]
    for threshold in parse_float_list(args.extra_thresholds):
        if np.isclose(threshold, submission_threshold):
            continue
        output_path = submission_path.with_name(
            f"{submission_path.stem}_thr{threshold_suffix(threshold)}{submission_path.suffix}"
        )
        submission_outputs.append(
            write_submission(
                sample_submission,
                test,
                test_probabilities,
                threshold,
                output_path,
            )
        )

    xgboost_probabilities = load_xgboost_test_probabilities(args.xgboost_artifact, test_x)
    blend_outputs = []
    if xgboost_probabilities is not None:
        for neural_weight in parse_float_list(args.blend_weights):
            blend_probabilities = (
                (1 - neural_weight) * xgboost_probabilities
                + neural_weight * test_probabilities
            )
            for threshold in [0.493, 0.495, 0.497]:
                output_path = submission_path.with_name(
                    f"{submission_path.stem}_blend_nn{threshold_suffix(neural_weight)}"
                    f"_thr{threshold_suffix(threshold)}{submission_path.suffix}"
                )
                blend_outputs.append(
                    write_submission(
                        sample_submission,
                        test,
                        blend_probabilities,
                        threshold,
                        output_path,
                    )
                )

    artifact_path = Path(args.artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "metrics": metrics,
            "feature_columns": feature_columns,
            "oof_probabilities": oof_probabilities,
            "test_probabilities": test_probabilities,
            "submission_outputs": submission_outputs,
            "blend_outputs": blend_outputs,
        },
        artifact_path,
    )
    if args.keep_models:
        artifact = joblib.load(artifact_path)
        artifact["model_states"] = model_states
        joblib.dump(artifact, artifact_path)

    print(json.dumps(metrics, indent=2))
    print("Wrote submissions:")
    for output in submission_outputs:
        print(json.dumps(output), flush=True)
    if blend_outputs:
        print("Wrote blend submissions:")
        for output in blend_outputs:
            print(json.dumps(output), flush=True)
    print(f"Wrote artifact: {artifact_path}")


if __name__ == "__main__":
    main()
