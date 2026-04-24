import json
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastprop.data import inverse_standard_scale, standard_scale
from rdkit import Chem
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
CHEMELEON_DIR = REPO_ROOT / "chemeleon"
if CHEMELEON_DIR.exists():
    sys.path.insert(0, str(CHEMELEON_DIR))

from tabpfn import TabPFNRegressor

from tabpfnmolprop.featurizer import create_chemeleon_fingerprinter

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PXR_SPLIT_MODE = os.getenv("PXR_SPLIT_MODE", "blinded").strip().lower()
_DEFAULT_BENCHMARK_SET = "openadmetpxr_internal" if PXR_SPLIT_MODE == "internal" else "openadmetpxr"
BENCHMARK_SET = os.getenv("BENCHMARK_SET", _DEFAULT_BENCHMARK_SET)
CHEMELEON_DEVICE = os.getenv("CHEMELEON_DEVICE", "cuda")
CHEMELEON_BATCH_SIZE = int(os.getenv("CHEMELEON_BATCH_SIZE", 256))
CV_N_SPLITS = int(os.getenv("CV_N_SPLITS", 5))
CV_RANDOM_SEED = int(os.getenv("CV_RANDOM_SEED", 42))
INTERNAL_SPLIT_DIR = Path(os.getenv("INTERNAL_SPLIT_DIR", "openadmet_pxr/internal_split"))
INTERNAL_TRAIN_CSV = Path(
    os.getenv("INTERNAL_TRAIN_CSV", str(INTERNAL_SPLIT_DIR / "internal_train.csv"))
)
INTERNAL_TEST_CSV = Path(
    os.getenv("INTERNAL_TEST_CSV", str(INTERNAL_SPLIT_DIR / "internal_test.csv"))
)

print(f"Running benchmark set {BENCHMARK_SET} with split mode {PXR_SPLIT_MODE}")


def resolve_chemeleon_device(requested_device: str) -> str:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested_device


def smiles_to_chemeleon_features(
    smiles_values: pd.Series,
    fingerprinter,
    batch_size: int,
) -> np.ndarray:
    valid_positions: list[int] = []
    valid_smiles: list[str] = []
    for idx, smiles in enumerate(smiles_values.tolist()):
        if not isinstance(smiles, str):
            continue
        if Chem.MolFromSmiles(smiles) is None:
            continue
        valid_positions.append(idx)
        valid_smiles.append(smiles)

    if not valid_smiles:
        raise RuntimeError("No valid SMILES available for CheMeleon fingerprinting.")

    batches = []
    bs = max(1, int(batch_size))
    for start in range(0, len(valid_smiles), bs):
        batch = valid_smiles[start : start + bs]
        batches.append(np.asarray(fingerprinter(batch), dtype=np.float32))
    valid_features = np.vstack(batches).astype(np.float32)

    features = np.zeros((len(smiles_values), valid_features.shape[1]), dtype=np.float32)
    features[np.asarray(valid_positions, dtype=np.int64)] = valid_features
    return features


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
) -> dict[str, object]:
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
    bundle: dict[str, object],
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
) -> dict[str, object]:
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32).ravel()
    splitter = KFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=CV_RANDOM_SEED)

    fold_rmses: list[float] = []
    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(features), start=1):
        fold_bundle = fit_scaled_tabpfn_regressor(
            features[train_idx], targets[train_idx], model_seed=model_seed
        )
        preds = predict_scaled_tabpfn_regressor(fold_bundle, features[valid_idx])
        fold_rmse = float(root_mean_squared_error(targets[valid_idx], preds))
        fold_rmses.append(fold_rmse)
        print(f"[CV] Fold {fold_idx}/{CV_N_SPLITS} RMSE: {fold_rmse:.4f}")

    rmse_mean = float(np.mean(fold_rmses))
    rmse_std = float(np.std(fold_rmses, ddof=1)) if len(fold_rmses) > 1 else 0.0
    print(f"[CV] Final {CV_N_SPLITS}-fold RMSE: {rmse_mean:.4f} +- {rmse_std:.4f}")
    return {
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
    pearson = float(np.corrcoef(truth, predictions)[0, 1]) if len(truth) > 1 else float("nan")
    return {
        "rmse": float(root_mean_squared_error(truth, predictions)),
        "mae": float(mean_absolute_error(truth, predictions)),
        "pearson": pearson,
    }


def load_split_frames(split_mode: str) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    if split_mode == "internal":
        train_source = str(INTERNAL_TRAIN_CSV)
        test_source = str(INTERNAL_TEST_CSV)
        return (
            pd.read_csv(INTERNAL_TRAIN_CSV),
            pd.read_csv(INTERNAL_TEST_CSV),
            train_source,
            test_source,
        )
    if split_mode == "blinded":
        train_source = "hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TRAIN.csv"
        test_source = "hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TEST_BLINDED.csv"
        return (
            pd.read_csv(train_source),
            pd.read_csv(test_source),
            train_source,
            test_source,
        )
    raise ValueError(
        f"Unsupported PXR_SPLIT_MODE={split_mode!r}. Use 'blinded' or 'internal'."
    )


def build_prediction_frame(
    split_mode: str,
    test_df: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    if split_mode == "internal":
        output_df = test_df.copy()
        output_df["pEC50_pred"] = predictions
        output_df["pEC50_error"] = output_df["pEC50_pred"] - output_df["pEC50"]
        return output_df

    return pd.DataFrame(
        {
            "SMILES": test_df["SMILES"],
            "Molecule Name": test_df["Molecule Name"],
            "pEC50": predictions,
        }
    )


if __name__ == "__main__":
    resolved_device = resolve_chemeleon_device(CHEMELEON_DEVICE)
    chemeleon_fingerprinter = create_chemeleon_fingerprinter(resolved_device)

    benchmark_name = f"{BENCHMARK_SET}"
    output_dir = Path("openadmet_pxr") / benchmark_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for random_seed in (42,):
        set_global_seed(random_seed)
        seed_dir = output_dir / f"seed_{random_seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        train_df, test_df, train_source, test_source = load_split_frames(PXR_SPLIT_MODE)

        train_df.to_csv(output_dir / "train.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)

        train_smiles = train_df["SMILES"]
        test_smiles = test_df["SMILES"]
        train_targets = train_df["pEC50"].to_numpy(dtype=np.float32).ravel()

        train_features = smiles_to_chemeleon_features(
            train_smiles, chemeleon_fingerprinter, CHEMELEON_BATCH_SIZE
        )
        test_features = smiles_to_chemeleon_features(
            test_smiles, chemeleon_fingerprinter, CHEMELEON_BATCH_SIZE
        )

        print(
            f"Running {CV_N_SPLITS}-fold random CV (split_seed={CV_RANDOM_SEED}, "
            f"model_seed={random_seed})"
        )
        cv_metrics = run_random_cv(train_features, train_targets, model_seed=random_seed)
        cv_df = pd.DataFrame(
            {
                "fold": np.arange(1, len(cv_metrics["fold_rmses"]) + 1),
                "rmse": cv_metrics["fold_rmses"],
            }
        )
        cv_df.to_csv(seed_dir / "cv_metrics.csv", index=False)
        with open(seed_dir / "cv_metrics.json", "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "benchmark": benchmark_name,
                    "seed": random_seed,
                    "split_mode": PXR_SPLIT_MODE,
                    "train_csv": train_source,
                    "test_csv": test_source,
                    "cv_n_splits": CV_N_SPLITS,
                    "cv_split_seed": CV_RANDOM_SEED,
                    "fold_rmses": cv_metrics["fold_rmses"],
                    "rmse_mean": cv_metrics["rmse_mean"],
                    "rmse_std": cv_metrics["rmse_std"],
                },
                fp,
                indent=2,
            )

        print(f"Starting final full-train evaluation on {PXR_SPLIT_MODE} test split.")
        final_bundle = fit_scaled_tabpfn_regressor(
            train_features, train_targets, model_seed=random_seed
        )
        predictions = predict_scaled_tabpfn_regressor(final_bundle, test_features)

        metrics_payload: dict[str, object] = {
            "benchmark": benchmark_name,
            "seed": random_seed,
            "split_mode": PXR_SPLIT_MODE,
            "train_csv": train_source,
            "test_csv": test_source,
            "chemeleon_device": resolved_device,
            "batch_size": CHEMELEON_BATCH_SIZE,
            "cv_n_splits": CV_N_SPLITS,
            "cv_split_seed": CV_RANDOM_SEED,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "feature_dim": int(train_features.shape[1]),
            "cv_metrics": cv_metrics,
        }

        if PXR_SPLIT_MODE == "internal":
            test_targets = test_df["pEC50"].to_numpy(dtype=np.float32).ravel()
            internal_test_metrics = evaluate_predictions(test_targets, predictions)
            metrics_payload["internal_test_metrics"] = internal_test_metrics
            print(
                f"Internal test RMSE={internal_test_metrics['rmse']:.4f} "
                f"| MAE={internal_test_metrics['mae']:.4f} "
                f"| Pearson={internal_test_metrics['pearson']:.4f}"
            )

        output_df = build_prediction_frame(PXR_SPLIT_MODE, test_df, predictions)
        output_df.to_csv(output_dir / "predictions.csv", index=False)
        output_df.to_csv(seed_dir / "predictions.csv", index=False)

        with open(seed_dir / "metrics.json", "w", encoding="utf-8") as fp:
            json.dump(metrics_payload, fp, indent=2)
