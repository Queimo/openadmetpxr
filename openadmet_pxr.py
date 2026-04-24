import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
CHEMELEON_DIR = REPO_ROOT / "chemeleon"
if CHEMELEON_DIR.exists():
    sys.path.insert(0, str(CHEMELEON_DIR))

from openadmet_pxr_eval import FeatureResult, run_openadmet_pxr_experiment
from tabpfnmolprop.featurizer import create_chemeleon_fingerprinter

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

BENCHMARK_SET = os.getenv("BENCHMARK_SET", "openadmetpxr")
CHEMELEON_DEVICE = os.getenv("CHEMELEON_DEVICE", "cuda")
CHEMELEON_BATCH_SIZE = int(os.getenv("CHEMELEON_BATCH_SIZE", 256))


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
        del split_name, seed, stage_dir
        return FeatureResult(
            train_features=smiles_to_chemeleon_features(
                train_df["SMILES"], chemeleon_fingerprinter, CHEMELEON_BATCH_SIZE
            ),
            test_features=smiles_to_chemeleon_features(
                test_df["SMILES"], chemeleon_fingerprinter, CHEMELEON_BATCH_SIZE
            ),
            metadata={"chemeleon_device": resolved_device},
        )

    run_openadmet_pxr_experiment(
        benchmark_name=BENCHMARK_SET,
        feature_name="chemeleon",
        build_features=build_features,
        feature_config={
            "requested_device": CHEMELEON_DEVICE,
            "resolved_device": resolved_device,
            "batch_size": CHEMELEON_BATCH_SIZE,
        },
    )
