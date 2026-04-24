import os
import sys
import warnings
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from openadmet_pxr_eval import FeatureResult, run_openadmet_pxr_experiment
from tabpfnmolprop.featurizer import compute_descriptors_from_smiles

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

BENCHMARK_SET = os.getenv("BENCHMARK_SET", "openadmetpxr_mordred")


if __name__ == "__main__":

    def build_features(
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        split_name: str,
        seed: int,
        stage_dir: Path,
    ) -> FeatureResult:
        del split_name, seed, stage_dir
        train_features = compute_descriptors_from_smiles(train_df["SMILES"])
        test_features = compute_descriptors_from_smiles(test_df["SMILES"])
        return FeatureResult(
            train_features=train_features,
            test_features=test_features,
            metadata={"descriptor_family": "mordred"},
        )

    run_openadmet_pxr_experiment(
        benchmark_name=BENCHMARK_SET,
        feature_name="mordred",
        build_features=build_features,
        feature_config={"descriptor_family": "mordred"},
    )
