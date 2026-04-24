from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from fastprop.data import inverse_standard_scale, standard_scale
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor

TARGET_COLUMN = "pEC50"
DEFAULT_OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", "openadmet_pxr"))
DEFAULT_CV_N_SPLITS = int(os.getenv("CV_N_SPLITS", 5))
DEFAULT_CV_RANDOM_SEED = int(os.getenv("CV_RANDOM_SEED", 42))
DEFAULT_SEEDS = tuple(
    int(seed.strip())
    for seed in os.getenv("RANDOM_SEEDS", "42").split(",")
    if seed.strip()
)
if not DEFAULT_SEEDS:
    DEFAULT_SEEDS = (42,)

INTERNAL_SPLIT_DIR = Path(os.getenv("INTERNAL_SPLIT_DIR", "openadmet_pxr/internal_split"))
INTERNAL_TRAIN_CSV = Path(
    os.getenv("INTERNAL_TRAIN_CSV", str(INTERNAL_SPLIT_DIR / "internal_train.csv"))
)
INTERNAL_TEST_CSV = Path(
    os.getenv("INTERNAL_TEST_CSV", str(INTERNAL_SPLIT_DIR / "internal_test.csv"))
)

PXR_HF_BASE = "hf://datasets/openadmet/pxr-challenge-train-test"
BLINDED_TRAIN_CSV = os.getenv(
    "BLINDED_TRAIN_CSV",
    f"{PXR_HF_BASE}/pxr-challenge_TRAIN.csv",
)
BLINDED_TEST_CSV = os.getenv(
    "BLINDED_TEST_CSV",
    f"{PXR_HF_BASE}/pxr-challenge_TEST_BLINDED.csv",
)
SPLIT_ORDER = ("internal", "blinded")


@dataclass
class FeatureResult:
    train_features: np.ndarray
    test_features: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    prediction_columns: dict[str, np.ndarray] = field(default_factory=dict)
    artifacts: dict[str, pd.DataFrame] = field(default_factory=dict)


FeatureBuilder = Callable[..., FeatureResult]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_scaled_tabpfn_regressor(
    features: np.ndarray,
    targets: np.ndarray,
    model_seed: int,
) -> dict[str, Any]:
    x_tensor = torch.tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32)
    y_tensor = torch.tensor(
        np.asarray(targets, dtype=np.float32).reshape(-1, 1), dtype=torch.float32
    )

    _, y_mean, y_var = standard_scale(y_tensor)
    y_scaled = standard_scale(y_tensor, y_mean, y_var).numpy().ravel()

    _, x_mean, x_var = standard_scale(x_tensor)
    x_scaled = standard_scale(x_tensor, x_mean, x_var).clamp(min=-6, max=6).numpy()

    model = TabPFNRegressor(random_state=model_seed, ignore_pretraining_limits=True)
    model.fit(x_scaled, y_scaled)
    return {
        "model": model,
        "x_mean": x_mean,
        "x_var": x_var,
        "y_mean": y_mean,
        "y_var": y_var,
    }


def predict_scaled_tabpfn_regressor(
    bundle: dict[str, Any],
    features: np.ndarray,
) -> np.ndarray:
    x_tensor = torch.tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32)
    x_scaled = (
        standard_scale(x_tensor, bundle["x_mean"], bundle["x_var"])
        .clamp(min=-6, max=6)
        .numpy()
    )
    pred_scaled = bundle["model"].predict(x_scaled).flatten()
    pred_tensor = torch.tensor(pred_scaled, dtype=torch.float32).reshape(-1, 1)
    predictions = inverse_standard_scale(
        pred_tensor, bundle["y_mean"], bundle["y_var"]
    ).numpy().ravel()
    return predictions.astype(np.float32)


def run_random_cv(
    features: np.ndarray,
    targets: np.ndarray,
    model_seed: int,
    n_splits: int = DEFAULT_CV_N_SPLITS,
    split_seed: int = DEFAULT_CV_RANDOM_SEED,
) -> dict[str, Any]:
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32).ravel()
    fold_count = min(int(n_splits), len(targets))
    if fold_count < 2:
        raise ValueError(f"Need at least 2 rows for CV, got {len(targets)}.")

    splitter = KFold(n_splits=fold_count, shuffle=True, random_state=split_seed)
    fold_metrics: list[dict[str, float]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(features), start=1):
        fold_bundle = fit_scaled_tabpfn_regressor(
            features[train_idx], targets[train_idx], model_seed=model_seed
        )
        preds = predict_scaled_tabpfn_regressor(fold_bundle, features[valid_idx])
        metrics = evaluate_predictions(targets[valid_idx], preds)
        fold_metrics.append({"fold": int(fold_idx), **metrics})
        print(
            f"[CV] Fold {fold_idx}/{fold_count} "
            f"RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} "
            f"Pearson={metrics['pearson']:.4f}"
        )

    fold_rmses = [fold["rmse"] for fold in fold_metrics]
    rmse_mean = float(np.mean(fold_rmses))
    rmse_std = float(np.std(fold_rmses, ddof=1)) if len(fold_rmses) > 1 else 0.0
    print(f"[CV] Final {fold_count}-fold RMSE: {rmse_mean:.4f} +- {rmse_std:.4f}")
    return {
        "n_splits": int(fold_count),
        "split_seed": int(split_seed),
        "fold_metrics": fold_metrics,
        "fold_rmses": [float(v) for v in fold_rmses],
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
    }


def evaluate_predictions(
    truth: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    truth = np.asarray(truth, dtype=np.float32).ravel()
    predictions = np.asarray(predictions, dtype=np.float32).ravel()
    pearson = (
        float(np.corrcoef(truth, predictions)[0, 1])
        if len(truth) > 1
        else float("nan")
    )
    return {
        "rmse": float(root_mean_squared_error(truth, predictions)),
        "mae": float(mean_absolute_error(truth, predictions)),
        "pearson": pearson,
    }


def load_split_frames(split_name: str) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    if split_name == "internal":
        return (
            pd.read_csv(INTERNAL_TRAIN_CSV),
            pd.read_csv(INTERNAL_TEST_CSV),
            str(INTERNAL_TRAIN_CSV),
            str(INTERNAL_TEST_CSV),
        )
    if split_name == "blinded":
        return (
            pd.read_csv(BLINDED_TRAIN_CSV),
            pd.read_csv(BLINDED_TEST_CSV),
            BLINDED_TRAIN_CSV,
            BLINDED_TEST_CSV,
        )
    raise ValueError(f"Unsupported split_name={split_name!r}.")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2)


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


class WandbLogger:
    def __init__(self, run_config: dict[str, Any]) -> None:
        self._wandb = None
        self._run = None
        self.enabled = _env_flag("WANDB_ENABLED", True)
        if not self.enabled:
            return
        try:
            import wandb
        except ImportError:
            print("[WANDB] wandb is not installed; skipping W&B logging.")
            return

        self._wandb = wandb
        try:
            self._run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "openadmet-pxr"),
                entity=os.getenv("WANDB_ENTITY") or None,
                group=str(run_config["experiment"]["benchmark"]),
                name=(
                    f"{run_config['experiment']['benchmark']}"
                    f"-seed{run_config['run']['seed']}"
                ),
                tags=[
                    "openadmet-pxr",
                    str(run_config["experiment"]["feature_name"]),
                ],
                config=_json_safe(run_config),
                reinit=True,
            )
        except Exception as exc:
            print(f"[WANDB] Could not initialize W&B logging: {exc}")
            self._wandb = None
            self._run = None

    @property
    def active(self) -> bool:
        return self._wandb is not None and self._run is not None

    def log_stage(
        self,
        split_name: str,
        metrics_payload: dict[str, Any],
        files: dict[str, Path],
    ) -> None:
        if not self.active:
            return

        flat_metrics: dict[str, float] = {}
        cv_metrics = metrics_payload.get("cv_metrics", {})
        if isinstance(cv_metrics, dict):
            flat_metrics[f"{split_name}/cv_rmse_mean"] = cv_metrics["rmse_mean"]
            flat_metrics[f"{split_name}/cv_rmse_std"] = cv_metrics["rmse_std"]
            fold_metrics = cv_metrics.get("fold_metrics")
            if isinstance(fold_metrics, list) and fold_metrics:
                fold_df = pd.DataFrame(fold_metrics)
                self._run.log(
                    {f"{split_name}/cv_fold_metrics": self._wandb.Table(dataframe=fold_df)}
                )

        test_metrics = metrics_payload.get("test_metrics")
        if isinstance(test_metrics, dict):
            for metric_name, metric_value in test_metrics.items():
                if metric_value is not None and np.isfinite(metric_value):
                    flat_metrics[f"{split_name}/test_{metric_name}"] = metric_value
        if flat_metrics:
            self._run.log(flat_metrics)

        stage_metrics_row = {
            "split": split_name,
            "seed": metrics_payload.get("seed"),
            "cv_rmse_mean": cv_metrics.get("rmse_mean")
            if isinstance(cv_metrics, dict)
            else None,
            "cv_rmse_std": cv_metrics.get("rmse_std")
            if isinstance(cv_metrics, dict)
            else None,
        }
        test_metrics = metrics_payload.get("test_metrics")
        if isinstance(test_metrics, dict):
            stage_metrics_row["test_rmse"] = test_metrics.get("rmse")
            stage_metrics_row["test_mae"] = test_metrics.get("mae")
            stage_metrics_row["test_pearson"] = test_metrics.get("pearson")
        self._run.log(
            {
                f"{split_name}/stage_metrics": self._wandb.Table(
                    dataframe=pd.DataFrame([stage_metrics_row])
                )
            }
        )

        predictions_path = files.get("raw_predictions")
        if predictions_path is not None and predictions_path.exists():
            prediction_df = pd.read_csv(predictions_path)
            self._run.log(
                {f"{split_name}/raw_predictions": self._wandb.Table(dataframe=prediction_df)}
            )

        artifact = self._wandb.Artifact(
            name=(
                f"{metrics_payload['benchmark']}-seed"
                f"{metrics_payload['seed']}-{split_name}"
            ),
            type="openadmet-pxr-eval",
            metadata=_json_safe(metrics_payload),
        )
        for path in files.values():
            if path.exists():
                artifact.add_file(str(path))
        self._run.log_artifact(artifact)

    def finish(self) -> None:
        if self.active:
            self._run.finish()


def _build_raw_prediction_frame(
    split_name: str,
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    prediction_columns: dict[str, np.ndarray],
) -> pd.DataFrame:
    raw_df = test_df.copy()
    raw_df["pEC50_pred"] = predictions
    if split_name == "internal" and TARGET_COLUMN in raw_df.columns:
        raw_df["pEC50_error"] = raw_df["pEC50_pred"] - raw_df[TARGET_COLUMN]
    for column_name, values in prediction_columns.items():
        raw_df[column_name] = np.asarray(values).ravel()
    return raw_df


def _build_submission_frame(test_df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SMILES": test_df["SMILES"],
            "Molecule Name": test_df["Molecule Name"],
            TARGET_COLUMN: predictions,
        }
    )


def _run_stage(
    *,
    benchmark_name: str,
    feature_name: str,
    feature_config: dict[str, Any],
    build_features: FeatureBuilder,
    seed: int,
    split_name: str,
    output_dir: Path,
    seed_dir: Path,
    cv_n_splits: int,
    cv_random_seed: int,
    wandb_logger: WandbLogger,
    run_config_path: Path,
) -> dict[str, Any]:
    stage_dir = seed_dir / split_name
    stage_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, train_source, test_source = load_split_frames(split_name)
    train_df.to_csv(stage_dir / "train.csv", index=False)
    test_df.to_csv(stage_dir / "test.csv", index=False)

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"Training split {train_source} is missing {TARGET_COLUMN!r}.")
    train_targets = train_df[TARGET_COLUMN].to_numpy(dtype=np.float32).ravel()

    print(
        f"[{split_name}] Building {feature_name} features for "
        f"{len(train_df)} train rows and {len(test_df)} test rows."
    )
    feature_result = build_features(
        train_df=train_df,
        test_df=test_df,
        split_name=split_name,
        seed=seed,
        stage_dir=stage_dir,
    )
    train_features = np.asarray(feature_result.train_features, dtype=np.float32)
    test_features = np.asarray(feature_result.test_features, dtype=np.float32)

    print(
        f"[{split_name}] Running {cv_n_splits}-fold random CV "
        f"(split_seed={cv_random_seed}, model_seed={seed})."
    )
    cv_metrics = run_random_cv(
        train_features,
        train_targets,
        model_seed=seed,
        n_splits=cv_n_splits,
        split_seed=cv_random_seed,
    )
    cv_df = pd.DataFrame(cv_metrics["fold_metrics"])
    cv_df.to_csv(stage_dir / "cv_metrics.csv", index=False)
    _write_json(stage_dir / "cv_metrics.json", cv_metrics)

    print(f"[{split_name}] Fitting final model on the full {split_name} train split.")
    final_bundle = fit_scaled_tabpfn_regressor(
        train_features, train_targets, model_seed=seed
    )
    predictions = predict_scaled_tabpfn_regressor(final_bundle, test_features)

    raw_prediction_df = _build_raw_prediction_frame(
        split_name, test_df, predictions, feature_result.prediction_columns
    )
    raw_prediction_path = stage_dir / "raw_predictions.csv"
    raw_prediction_df.to_csv(raw_prediction_path, index=False)

    if split_name == "blinded":
        prediction_df = _build_submission_frame(test_df, predictions)
    else:
        prediction_df = raw_prediction_df
    prediction_path = stage_dir / "predictions.csv"
    prediction_df.to_csv(prediction_path, index=False)

    artifact_paths: dict[str, Path] = {
        "run_config_json": run_config_path,
        "train": stage_dir / "train.csv",
        "test": stage_dir / "test.csv",
        "cv_metrics_csv": stage_dir / "cv_metrics.csv",
        "cv_metrics_json": stage_dir / "cv_metrics.json",
        "predictions": prediction_path,
        "raw_predictions": raw_prediction_path,
    }
    for artifact_name, artifact_df in feature_result.artifacts.items():
        artifact_path = stage_dir / artifact_name
        artifact_df.to_csv(artifact_path, index=False)
        artifact_paths[artifact_name] = artifact_path

    metrics_payload: dict[str, Any] = {
        "benchmark": benchmark_name,
        "seed": int(seed),
        "split": split_name,
        "train_csv": train_source,
        "test_csv": test_source,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "target_column": TARGET_COLUMN,
        "feature_name": feature_name,
        "feature_dim": int(train_features.shape[1]),
        "feature_config": feature_config,
        "feature_metadata": feature_result.metadata,
        "model": {
            "name": "TabPFNRegressor",
            "random_state": int(seed),
            "ignore_pretraining_limits": True,
            "feature_standardization": "fastprop.standard_scale",
            "target_standardization": "fastprop.standard_scale",
            "feature_clamp": [-6, 6],
        },
        "cv_metrics": cv_metrics,
    }

    if split_name == "internal":
        test_targets = test_df[TARGET_COLUMN].to_numpy(dtype=np.float32).ravel()
        test_metrics = evaluate_predictions(test_targets, predictions)
        metrics_payload["test_metrics"] = test_metrics
        print(
            f"[internal] Test RMSE={test_metrics['rmse']:.4f} "
            f"MAE={test_metrics['mae']:.4f} "
            f"Pearson={test_metrics['pearson']:.4f}"
        )
    else:
        metrics_payload["test_metrics"] = None
        print("[blinded] Wrote blinded predictions; labels are not available for metrics.")

    metrics_path = stage_dir / "metrics.json"
    _write_json(metrics_path, metrics_payload)
    artifact_paths["metrics_json"] = metrics_path

    if split_name == "blinded":
        prediction_df.to_csv(output_dir / "predictions.csv", index=False)
        prediction_df.to_csv(seed_dir / "predictions.csv", index=False)
        raw_prediction_df.to_csv(seed_dir / "raw_predictions.csv", index=False)
        _write_json(seed_dir / "metrics.json", metrics_payload)
        train_df.to_csv(output_dir / "train.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)
    else:
        raw_prediction_df.to_csv(seed_dir / "internal_predictions.csv", index=False)

    wandb_logger.log_stage(split_name, metrics_payload, artifact_paths)
    return metrics_payload


def run_openadmet_pxr_experiment(
    *,
    benchmark_name: str,
    feature_name: str,
    build_features: FeatureBuilder,
    feature_config: dict[str, Any] | None = None,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    cv_n_splits: int = DEFAULT_CV_N_SPLITS,
    cv_random_seed: int = DEFAULT_CV_RANDOM_SEED,
) -> None:
    feature_config = feature_config or {}
    output_dir = output_root / benchmark_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running benchmark {benchmark_name} with feature set {feature_name}")

    for seed in seeds:
        set_global_seed(seed)
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        run_config = {
            "experiment": {
                "benchmark": benchmark_name,
                "feature_name": feature_name,
            },
            "run": {
                "seed": int(seed),
                "split_order": list(SPLIT_ORDER),
                "output_dir": str(output_dir),
                "seed_dir": str(seed_dir),
            },
            "data": {
                "target_column": TARGET_COLUMN,
                "internal_train_csv": str(INTERNAL_TRAIN_CSV),
                "internal_test_csv": str(INTERNAL_TEST_CSV),
                "blinded_train_csv": BLINDED_TRAIN_CSV,
                "blinded_test_csv": BLINDED_TEST_CSV,
            },
            "stages": {
                "internal": {
                    "train_csv": str(INTERNAL_TRAIN_CSV),
                    "test_csv": str(INTERNAL_TEST_CSV),
                    "has_test_labels": True,
                },
                "blinded": {
                    "train_csv": BLINDED_TRAIN_CSV,
                    "test_csv": BLINDED_TEST_CSV,
                    "has_test_labels": False,
                },
            },
            "cv": {
                "n_splits": int(cv_n_splits),
                "split_seed": int(cv_random_seed),
            },
            "model": {
                "name": "TabPFNRegressor",
                "ignore_pretraining_limits": True,
                "feature_standardization": "fastprop.standard_scale",
                "target_standardization": "fastprop.standard_scale",
                "feature_clamp": [-6, 6],
            },
            "feature": feature_config,
        }
        run_config_path = seed_dir / "run_config.json"
        _write_json(run_config_path, run_config)

        wandb_logger = WandbLogger(run_config)
        try:
            summaries = []
            for split_name in SPLIT_ORDER:
                summaries.append(
                    _run_stage(
                        benchmark_name=benchmark_name,
                        feature_name=feature_name,
                        feature_config=feature_config,
                        build_features=build_features,
                        seed=seed,
                        split_name=split_name,
                        output_dir=output_dir,
                        seed_dir=seed_dir,
                        cv_n_splits=cv_n_splits,
                        cv_random_seed=cv_random_seed,
                        wandb_logger=wandb_logger,
                        run_config_path=run_config_path,
                    )
                )
            _write_json(seed_dir / "summary.json", {"stages": summaries})
        finally:
            wandb_logger.finish()
