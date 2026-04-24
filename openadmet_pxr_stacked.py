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

BENCHMARK_SET = os.getenv("BENCHMARK_SET", "openadmetpxr_stacked")
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

# Chosen from the available assay labels as the strongest non-error correlates of pEC50
# on the internal split. The std.error columns were excluded because they are likely to
# encode assay noise more than stable structure-driven signal.
AUXILIARY_TARGET_COLUMNS = [
    "pEC50_ci.lower (-log10(molarity))",
    "pEC50_ci.upper (-log10(molarity))",
    # "Emax_ci.upper (log2FC vs. baseline)",
    # "Emax.vs.pos.ctrl_ci.upper (dimensionless)",
]

print(f"Running benchmark set {BENCHMARK_SET}")


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


def build_auxiliary_stack(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_df: pd.DataFrame,
    model_seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    n_train = train_features.shape[0]
    train_aux_features: list[np.ndarray] = []
    test_aux_features: list[np.ndarray] = []
    summaries: list[dict[str, object]] = []

    for target_column in AUXILIARY_TARGET_COLUMNS:
        target_values = pd.to_numeric(train_df[target_column], errors="coerce").to_numpy(
            dtype=np.float32
        )
        available_idx = np.flatnonzero(np.isfinite(target_values))
        if len(available_idx) < 2:
            raise ValueError(
                f"Auxiliary label {target_column!r} has only {len(available_idx)} usable rows."
            )

        oof_predictions = np.full(n_train, np.nan, dtype=np.float32)
        fold_count = min(CV_N_SPLITS, len(available_idx))
        if fold_count < 2:
            raise ValueError(
                f"Need at least 2 usable rows to stack auxiliary label {target_column!r}."
            )
        splitter = KFold(n_splits=fold_count, shuffle=True, random_state=CV_RANDOM_SEED)

        for fold_idx, (fit_pos, valid_pos) in enumerate(
            splitter.split(available_idx), start=1
        ):
            fit_idx = available_idx[fit_pos]
            valid_idx = available_idx[valid_pos]
            fold_bundle = fit_scaled_tabpfn_regressor(
                train_features[fit_idx], target_values[fit_idx], model_seed=model_seed
            )
            oof_predictions[valid_idx] = predict_scaled_tabpfn_regressor(
                fold_bundle, train_features[valid_idx]
            )
            print(
                f"[STACK] {target_column} fold {fold_idx}/{fold_count} completed "
                f"({len(valid_idx)} OOF rows)."
            )

        full_bundle = fit_scaled_tabpfn_regressor(
            train_features[available_idx], target_values[available_idx], model_seed=model_seed
        )
        full_train_predictions = predict_scaled_tabpfn_regressor(full_bundle, train_features)
        missing_oof_mask = ~np.isfinite(oof_predictions)
        if missing_oof_mask.any():
            oof_predictions[missing_oof_mask] = full_train_predictions[missing_oof_mask]

        test_predictions = predict_scaled_tabpfn_regressor(full_bundle, test_features)
        train_aux_features.append(oof_predictions.reshape(-1, 1))
        test_aux_features.append(test_predictions.reshape(-1, 1))

        oof_target_values = target_values[available_idx]
        oof_predictions_available = oof_predictions[available_idx]
        oof_rmse = float(root_mean_squared_error(oof_target_values, oof_predictions_available))
        if len(oof_target_values) > 1:
            oof_pearson = float(np.corrcoef(oof_target_values, oof_predictions_available)[0, 1])
        else:
            oof_pearson = float("nan")
        summaries.append(
            {
                "target": target_column,
                "n_train_available": int(len(available_idx)),
                "oof_rmse": oof_rmse,
                "oof_pearson": oof_pearson,
            }
        )
        print(
            f"[STACK] {target_column} OOF RMSE={oof_rmse:.4f} "
            f"| OOF Pearson={oof_pearson:.4f}"
        )

    return (
        np.hstack(train_aux_features).astype(np.float32),
        np.hstack(test_aux_features).astype(np.float32),
        summaries,
    )


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

        train_df = pd.read_csv(INTERNAL_TRAIN_CSV)
        test_df = pd.read_csv(INTERNAL_TEST_CSV)

        train_df.to_csv(output_dir / "train.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)

        train_smiles = train_df["SMILES"]
        test_smiles = test_df["SMILES"]
        train_targets = train_df["pEC50"].to_numpy(dtype=np.float32).ravel()
        test_targets = test_df["pEC50"].to_numpy(dtype=np.float32).ravel()

        train_features = smiles_to_chemeleon_features(
            train_smiles, chemeleon_fingerprinter, CHEMELEON_BATCH_SIZE
        )
        test_features = smiles_to_chemeleon_features(
            test_smiles, chemeleon_fingerprinter, CHEMELEON_BATCH_SIZE
        )

        train_aux_features, test_aux_features, auxiliary_metrics = build_auxiliary_stack(
            train_features=train_features,
            test_features=test_features,
            train_df=train_df,
            model_seed=random_seed,
        )
        stacked_train_features = np.hstack(
            [train_features, train_aux_features]
        ).astype(np.float32)
        stacked_test_features = np.hstack([test_features, test_aux_features]).astype(np.float32)

        final_bundle = fit_scaled_tabpfn_regressor(
            stacked_train_features, train_targets, model_seed=random_seed
        )
        predictions = predict_scaled_tabpfn_regressor(final_bundle, stacked_test_features)
        test_metrics = evaluate_predictions(test_targets, predictions)

        output_df = test_df.copy()
        output_df["pEC50_pred"] = predictions
        output_df["pEC50_error"] = output_df["pEC50_pred"] - output_df["pEC50"]
        for aux_idx, aux_column in enumerate(AUXILIARY_TARGET_COLUMNS):
            output_df[f"stacked_pred::{aux_column}"] = test_aux_features[:, aux_idx]
        output_df.to_csv(output_dir / "predictions.csv", index=False)
        output_df.to_csv(seed_dir / "predictions.csv", index=False)

        train_aux_df = train_df[["Molecule Name", "SMILES", "pEC50"]].copy()
        test_aux_df = test_df[["Molecule Name", "SMILES", "pEC50"]].copy()
        for aux_idx, aux_column in enumerate(AUXILIARY_TARGET_COLUMNS):
            train_aux_df[f"stacked_oof::{aux_column}"] = train_aux_features[:, aux_idx]
            test_aux_df[f"stacked_pred::{aux_column}"] = test_aux_features[:, aux_idx]
        train_aux_df.to_csv(seed_dir / "stacked_train_aux_features.csv", index=False)
        test_aux_df.to_csv(seed_dir / "stacked_test_aux_features.csv", index=False)

        with open(seed_dir / "metrics.json", "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "benchmark": benchmark_name,
                    "seed": random_seed,
                    "train_csv": str(INTERNAL_TRAIN_CSV),
                    "test_csv": str(INTERNAL_TEST_CSV),
                    "chemeleon_device": resolved_device,
                    "batch_size": CHEMELEON_BATCH_SIZE,
                    "cv_n_splits": CV_N_SPLITS,
                    "cv_split_seed": CV_RANDOM_SEED,
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "base_feature_dim": int(train_features.shape[1]),
                    "stacked_feature_dim": int(stacked_train_features.shape[1]),
                    "auxiliary_targets": AUXILIARY_TARGET_COLUMNS,
                    "auxiliary_oof_metrics": auxiliary_metrics,
                    "internal_test_metrics": test_metrics,
                },
                fp,
                indent=2,
            )

        print(
            f"Internal test RMSE={test_metrics['rmse']:.4f} "
            f"| MAE={test_metrics['mae']:.4f} "
            f"| Pearson={test_metrics['pearson']:.4f}"
        )
