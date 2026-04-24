import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
CHEMELEON_DIR = REPO_ROOT / "chemeleon"
if CHEMELEON_DIR.exists():
    sys.path.insert(0, str(CHEMELEON_DIR))

from openadmet_pxr_eval import (
    DEFAULT_CV_N_SPLITS,
    DEFAULT_CV_RANDOM_SEED,
    FeatureResult,
    fit_scaled_tabpfn_regressor,
    predict_scaled_tabpfn_regressor,
    run_openadmet_pxr_experiment,
)
from tabpfnmolprop.featurizer import create_chemeleon_fingerprinter

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

BENCHMARK_SET = os.getenv("BENCHMARK_SET", "openadmetpxr_stacked")
CHEMELEON_DEVICE = os.getenv("CHEMELEON_DEVICE", "cuda")
CHEMELEON_BATCH_SIZE = int(os.getenv("CHEMELEON_BATCH_SIZE", 256))

# Chosen from the available assay labels as the strongest non-error correlates of pEC50
# on the internal split. The std.error columns were excluded because they are likely to
# encode assay noise more than stable structure-driven signal.
AUXILIARY_TARGET_COLUMNS = [
    "pEC50_ci.lower (-log10(molarity))",
    "pEC50_ci.upper (-log10(molarity))",
]


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
    resolved_batch_size = max(1, int(batch_size))
    for start in range(0, len(valid_smiles), resolved_batch_size):
        batch = valid_smiles[start : start + resolved_batch_size]
        batches.append(np.asarray(fingerprinter(batch), dtype=np.float32))
    valid_features = np.vstack(batches).astype(np.float32)

    features = np.zeros((len(smiles_values), valid_features.shape[1]), dtype=np.float32)
    features[np.asarray(valid_positions, dtype=np.int64)] = valid_features
    return features


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
                f"Auxiliary label {target_column!r} has only "
                f"{len(available_idx)} usable rows."
            )

        oof_predictions = np.full(n_train, np.nan, dtype=np.float32)
        fold_count = min(DEFAULT_CV_N_SPLITS, len(available_idx))
        splitter = KFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=DEFAULT_CV_RANDOM_SEED,
        )

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
            train_features[available_idx],
            target_values[available_idx],
            model_seed=model_seed,
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
        oof_rmse = float(
            root_mean_squared_error(oof_target_values, oof_predictions_available)
        )
        oof_pearson = (
            float(np.corrcoef(oof_target_values, oof_predictions_available)[0, 1])
            if len(oof_target_values) > 1
            else float("nan")
        )
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
            f"OOF Pearson={oof_pearson:.4f}"
        )

    return (
        np.hstack(train_aux_features).astype(np.float32),
        np.hstack(test_aux_features).astype(np.float32),
        summaries,
    )


def identity_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ["Molecule Name", "SMILES", "pEC50"] if column in df]
    return df[columns].copy()


if __name__ == "__main__":
    resolved_device = resolve_chemeleon_device(CHEMELEON_DEVICE)
    chemeleon_fingerprinter = create_chemeleon_fingerprinter(resolved_device)

    def build_features(
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        split_name: str,
        seed: int,
        stage_dir: Path,
    ) -> FeatureResult:
        del split_name, stage_dir
        train_features = smiles_to_chemeleon_features(
            train_df["SMILES"], chemeleon_fingerprinter, CHEMELEON_BATCH_SIZE
        )
        test_features = smiles_to_chemeleon_features(
            test_df["SMILES"], chemeleon_fingerprinter, CHEMELEON_BATCH_SIZE
        )
        train_aux, test_aux, auxiliary_metrics = build_auxiliary_stack(
            train_features=train_features,
            test_features=test_features,
            train_df=train_df,
            model_seed=seed,
        )
        stacked_train = np.hstack([train_features, train_aux]).astype(np.float32)
        stacked_test = np.hstack([test_features, test_aux]).astype(np.float32)

        train_aux_df = identity_frame(train_df)
        test_aux_df = identity_frame(test_df)
        prediction_columns = {}
        for aux_idx, aux_column in enumerate(AUXILIARY_TARGET_COLUMNS):
            train_aux_df[f"stacked_oof::{aux_column}"] = train_aux[:, aux_idx]
            test_aux_df[f"stacked_pred::{aux_column}"] = test_aux[:, aux_idx]
            prediction_columns[f"stacked_pred::{aux_column}"] = test_aux[:, aux_idx]

        return FeatureResult(
            train_features=stacked_train,
            test_features=stacked_test,
            metadata={
                "chemeleon_device": resolved_device,
                "base_feature_dim": int(train_features.shape[1]),
                "stacked_feature_dim": int(stacked_train.shape[1]),
                "auxiliary_targets": AUXILIARY_TARGET_COLUMNS,
                "auxiliary_oof_metrics": auxiliary_metrics,
            },
            prediction_columns=prediction_columns,
            artifacts={
                "stacked_train_aux_features.csv": train_aux_df,
                "stacked_test_aux_features.csv": test_aux_df,
            },
        )

    run_openadmet_pxr_experiment(
        benchmark_name=BENCHMARK_SET,
        feature_name="chemeleon_stacked_auxiliary",
        build_features=build_features,
        feature_config={
            "base_featurizer": "chemeleon",
            "requested_device": CHEMELEON_DEVICE,
            "resolved_device": resolved_device,
            "batch_size": CHEMELEON_BATCH_SIZE,
            "auxiliary_targets": AUXILIARY_TARGET_COLUMNS,
        },
    )
