import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from openadmet_pxr_eval import FeatureResult, run_openadmet_pxr_experiment

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
    return model, str(resolved_device)


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


if __name__ == "__main__":
    clamp_encoder, resolved_device = create_clamp_encoder(CLAMP_DEVICE)

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
            train_features=smiles_to_clamp_features(
                train_df["SMILES"], clamp_encoder, CLAMP_BATCH_SIZE
            ),
            test_features=smiles_to_clamp_features(
                test_df["SMILES"], clamp_encoder, CLAMP_BATCH_SIZE
            ),
            metadata={"clamp_device": resolved_device},
        )

    run_openadmet_pxr_experiment(
        benchmark_name=BENCHMARK_SET,
        feature_name="clamp",
        build_features=build_features,
        feature_config={
            "requested_device": CLAMP_DEVICE,
            "resolved_device": resolved_device,
            "batch_size": CLAMP_BATCH_SIZE,
        },
    )
