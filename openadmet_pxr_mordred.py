import pandas as pd


import datetime
import json
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polaris as po
import torch
from fastprop.data import inverse_standard_scale, standard_scale
from polaris.utils.types import TargetType
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from tabpfnmolprop.featurizer import compute_descriptors_from_smiles



REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
CHEMELEON_DIR = REPO_ROOT / "chemeleon"
if CHEMELEON_DIR.exists():
    sys.path.insert(0, str(CHEMELEON_DIR))

from tabpfn import TabPFNClassifier, TabPFNRegressor


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

BENCHMARK_SET = os.getenv("BENCHMARK_SET", "openadmetpxr_mordred")
CHEMELEON_DEVICE = os.getenv("CHEMELEON_DEVICE", "cuda")
CHEMELEON_BATCH_SIZE = int(os.getenv("CHEMELEON_BATCH_SIZE", 256))
CV_N_SPLITS = int(os.getenv("CV_N_SPLITS", 5))
CV_RANDOM_SEED = int(os.getenv("CV_RANDOM_SEED", 42))
print(f"Running benchmark set {BENCHMARK_SET}")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        x_train = torch.tensor(features[train_idx], dtype=torch.float32)
        x_valid = torch.tensor(features[valid_idx], dtype=torch.float32)
        y_train = torch.tensor(targets[train_idx], dtype=torch.float32).reshape(-1, 1)
        y_valid = targets[valid_idx]

        _, y_mean, y_var = standard_scale(y_train)
        y_train_scaled = standard_scale(y_train, y_mean, y_var).numpy().ravel()

        _, x_mean, x_var = standard_scale(x_train)
        x_train_scaled = (
            standard_scale(x_train, x_mean, x_var).clamp(min=-6, max=6).numpy()
        )
        x_valid_scaled = (
            standard_scale(x_valid, x_mean, x_var).clamp(min=-6, max=6).numpy()
        )

        fold_model = TabPFNRegressor(
            random_state=model_seed, ignore_pretraining_limits=True
        )
        fold_model.fit(x_train_scaled, y_train_scaled)
        pred_scaled = fold_model.predict(x_valid_scaled).flatten()
        pred_tensor = torch.tensor(pred_scaled, dtype=torch.float32).reshape(-1, 1)
        preds = inverse_standard_scale(pred_tensor, y_mean, y_var).numpy().ravel()

        fold_rmse = float(root_mean_squared_error(y_valid, preds))
        fold_rmses.append(fold_rmse)
        print(f"[CV] Fold {fold_idx}/{CV_N_SPLITS} RMSE: {fold_rmse:.4f}")

    rmse_mean = float(np.mean(fold_rmses))
    rmse_std = float(np.std(fold_rmses, ddof=1)) if len(fold_rmses) > 1 else 0.0
    print(f"[CV] Final {CV_N_SPLITS}-fold RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")
    return {
        "fold_rmses": [float(v) for v in fold_rmses],
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
    }


if __name__ == "__main__":
    benchmark_name = f"{BENCHMARK_SET}"
    output_dir = Path("openadmet_pxr") / benchmark_name
    output_dir.mkdir(parents=True, exist_ok=True)
    # for random_seed in (42, 117, 709, 1701, 9001):
    for random_seed in (42,):
        set_global_seed(random_seed)
        seed_dir = output_dir / f"seed_{random_seed}"
        if not seed_dir.exists():
            seed_dir.mkdir()
        
        splits = {'train': 'pxr-challenge_TRAIN.csv', 'test': 'pxr-challenge_TEST_BLINDED.csv'}
        train_df = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/" + splits["train"])
        test_df = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/" + splits["test"])
        train_smiles = train_df["SMILES"]
        test_smiles = test_df["SMILES"]
        raw_targets = train_df["pEC50"].to_numpy().ravel()

        #save the train and test data for reference
        train_df.to_csv(output_dir / "train.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)

        train_desc = compute_descriptors_from_smiles(train_smiles)
        test_desc = compute_descriptors_from_smiles(test_smiles)

        print(
            f"Running {CV_N_SPLITS}-fold random CV (split_seed={CV_RANDOM_SEED}, model_seed={random_seed})"
        )
        cv_metrics = run_random_cv(train_desc, raw_targets, model_seed=random_seed)
        print(
            f"CV summary before full evaluation: RMSE={cv_metrics['rmse_mean']:.4f} ± {cv_metrics['rmse_std']:.4f}"
        )
        cv_df = pd.DataFrame(
            {"fold": np.arange(1, len(cv_metrics["fold_rmses"]) + 1), "rmse": cv_metrics["fold_rmses"]}
        )
        cv_df.to_csv(seed_dir / "cv_metrics.csv", index=False)
        with open(seed_dir / "cv_metrics.json", "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "benchmark": benchmark_name,
                    "seed": random_seed,
                    "cv_n_splits": CV_N_SPLITS,
                    "cv_split_seed": CV_RANDOM_SEED,
                    "fold_rmses": cv_metrics["fold_rmses"],
                    "rmse_mean": cv_metrics["rmse_mean"],
                    "rmse_std": cv_metrics["rmse_std"],
                },
                fp,
                indent=2,
            )
        print("Starting final full-train evaluation on blinded test set.")

        # Convert to torch tensors for use with standard_scale
        train_features_tensor = torch.tensor(train_desc, dtype=torch.float32)
        targets_tensor = torch.tensor(raw_targets, dtype=torch.float32).reshape(-1, 1)

        _, target_means, target_vars = standard_scale(targets_tensor)
        targets_scaled = (
            standard_scale(targets_tensor, target_means, target_vars)
            .numpy()
            .ravel()
        )
        targets = targets_scaled

        # Feature scaling - exactly as in the MLP script with clamping as in RescalingEncoder
        _, feature_means, feature_vars = standard_scale(train_features_tensor)
        train_features = (
            standard_scale(train_features_tensor, feature_means, feature_vars)
            .clamp(min=-6, max=6)
            .numpy()
        )
        test_features_tensor = torch.tensor(test_desc, dtype=torch.float32)
        test_features = (
            standard_scale(test_features_tensor, feature_means, feature_vars)
            .clamp(min=-6, max=6)
            .numpy()
        )

        off_load_dir = seed_dir / benchmark_name
        off_load_dir.mkdir(parents=True, exist_ok=True)

        # Create and train model
        model = TabPFNRegressor(random_state=random_seed, ignore_pretraining_limits=True)

        # Train model directly on the fingerprints
        model.fit(train_features, targets)

        # generate predictions and evaluate performance
        print(f"Evaluating benchmark: {benchmark_name}")

        # Get value predictions for regression tasks
        predictions = model.predict(test_features).flatten()
        # Inverse transform the predictions using the same scaling approach as in the MLP script
        predictions_tensor = torch.tensor(
            predictions, dtype=torch.float32
        ).reshape(-1, 1)
        predictions = (
            inverse_standard_scale(
                predictions_tensor, target_means, target_vars
            )
            .numpy()
            .ravel()
        )
        #csv
        # SMILES,Molecule Name,pEC50
        # CCO,OADMET-00000,6.23
        # c1ccccc1,OADMET-00001,5.87
        # ...
        output_df = pd.DataFrame({
            "SMILES": test_smiles,
            "Molecule Name": test_df["Molecule Name"],
            "pEC50": predictions
        })
        output_df.to_csv(output_dir / "predictions.csv", index=False)
