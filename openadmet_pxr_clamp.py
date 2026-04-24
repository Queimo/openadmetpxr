import os
import random
import sys
import warnings
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from fastprop.data import inverse_standard_scale, standard_scale
from rdkit import Chem
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor

OPENADMET_DIR = Path(__file__).resolve().parent
LOCAL_CLAMP_REPO = OPENADMET_DIR / "clamp"
if LOCAL_CLAMP_REPO.exists():
    # Ensure `import clamp` resolves to the package in `openadmet/clamp/clamp`.
    sys.path.insert(0, str(LOCAL_CLAMP_REPO))

import clamp

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

BENCHMARK_SET = os.getenv("BENCHMARK_SET", "openadmetpxr_clamp")
CLAMP_DEVICE = os.getenv("CLAMP_DEVICE", "cuda")
CLAMP_BATCH_SIZE = int(os.getenv("CLAMP_BATCH_SIZE", 256))
CV_N_SPLITS = int(os.getenv("CV_N_SPLITS", 5))
CV_RANDOM_SEED = int(os.getenv("CV_RANDOM_SEED", 42))
print(f"Running benchmark set {BENCHMARK_SET}")


def create_clamp_encoder(device: str):
    requested = device
    if device.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    resolved_device = torch.device(requested)

    model = clamp.CLAMP(device=str(resolved_device))
    model.to(resolved_device)
    model.device = str(resolved_device)
    model.eval()
    print(f"Using CLAMP device: {resolved_device}")
    return model


def smiles_to_clamp_features(
    smiles_values: pd.Series,
    model,
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
        raise RuntimeError("No valid SMILES available for CLAMP encoding.")

    batches = []
    bs = max(1, int(batch_size))
    for start in range(0, len(valid_smiles), bs):
        batch = valid_smiles[start : start + bs]
        with torch.no_grad():
            batch_embeddings = model.encode_smiles(batch).detach().cpu().numpy()
        batches.append(np.asarray(batch_embeddings, dtype=np.float32))
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
    clamp_encoder = create_clamp_encoder(CLAMP_DEVICE)
    benchmark_name = f"{BENCHMARK_SET}"
    output_dir = Path("openadmet_pxr") / benchmark_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # for random_seed in (42, 117, 709, 1701, 9001):
    for random_seed in (42,):
        set_global_seed(random_seed)
        seed_dir = output_dir / f"seed_{random_seed}"
        if not seed_dir.exists():
            seed_dir.mkdir()

        splits = {
            "train": "pxr-challenge_TRAIN.csv",
            "test": "pxr-challenge_TEST_BLINDED.csv",
        }
        train_df = pd.read_csv(
            "hf://datasets/openadmet/pxr-challenge-train-test/" + splits["train"]
        )
        test_df = pd.read_csv(
            "hf://datasets/openadmet/pxr-challenge-train-test/" + splits["test"]
        )

        # Save train/test data for reference.
        train_df.to_csv(output_dir / "train.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)

        train_smiles = train_df["SMILES"]
        test_smiles = test_df["SMILES"]
        raw_targets = train_df["pEC50"].to_numpy().ravel()

        # Compute frozen CLAMP embeddings from SMILES.
        train_features = smiles_to_clamp_features(
            train_smiles, clamp_encoder, CLAMP_BATCH_SIZE
        )
        test_features = smiles_to_clamp_features(
            test_smiles, clamp_encoder, CLAMP_BATCH_SIZE
        )

        print(
            f"Running {CV_N_SPLITS}-fold random CV (split_seed={CV_RANDOM_SEED}, model_seed={random_seed})"
        )
        cv_metrics = run_random_cv(train_features, raw_targets, model_seed=random_seed)
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

        train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
        targets_tensor = torch.tensor(raw_targets, dtype=torch.float32).reshape(-1, 1)

        _, target_means, target_vars = standard_scale(targets_tensor)
        targets_scaled = (
            standard_scale(targets_tensor, target_means, target_vars).numpy().ravel()
        )
        targets = targets_scaled

        _, feature_means, feature_vars = standard_scale(train_features_tensor)
        train_features = (
            standard_scale(train_features_tensor, feature_means, feature_vars)
            .clamp(min=-6, max=6)
            .numpy()
        )
        test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
        test_features = (
            standard_scale(test_features_tensor, feature_means, feature_vars)
            .clamp(min=-6, max=6)
            .numpy()
        )

        off_load_dir = seed_dir / benchmark_name
        off_load_dir.mkdir(parents=True, exist_ok=True)

        model = TabPFNRegressor(random_state=random_seed, ignore_pretraining_limits=True)
        model.fit(train_features, targets)

        print(f"Evaluating benchmark: {benchmark_name}")
        predictions = model.predict(test_features).flatten()

        predictions_tensor = torch.tensor(predictions, dtype=torch.float32).reshape(-1, 1)
        predictions = (
            inverse_standard_scale(predictions_tensor, target_means, target_vars)
            .numpy()
            .ravel()
        )

        output_df = pd.DataFrame(
            {
                "SMILES": test_smiles,
                "Molecule Name": test_df["Molecule Name"],
                "pEC50": predictions,
            }
        )
        output_df.to_csv(output_dir / "predictions.csv", index=False)
