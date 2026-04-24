import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
CHEMELEON_DIR = REPO_ROOT / "chemeleon"
if CHEMELEON_DIR.exists():
    sys.path.insert(0, str(CHEMELEON_DIR))
TABPFNMOLPROP_ALT_DIR = REPO_ROOT / "tabpfn-molprop"
if TABPFNMOLPROP_ALT_DIR.exists():
    sys.path.insert(0, str(TABPFNMOLPROP_ALT_DIR))

from tabpfnmolprop.featurizer import create_chemeleon_fingerprinter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an internal train/test split by greedily matching each unique "
            "blinded test molecule to the closest available training molecule in "
            "CheMeleon embedding space."
        )
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("openadmet_pxr/openadmetpxr/train.csv"),
        help="Path to the labeled training CSV.",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("openadmet_pxr/openadmetpxr/test.csv"),
        help="Path to the blinded test CSV used as the matching anchor set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("openadmet_pxr/internal_split"),
        help="Directory where the internal split files are written.",
    )
    parser.add_argument(
        "--chemeleon-device",
        type=str,
        default=os.getenv("CHEMELEON_DEVICE", "cuda"),
        help="Device passed to the CheMeleon encoder.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("CHEMELEON_BATCH_SIZE", 256)),
        help="Batch size used for CheMeleon embeddings.",
    )
    return parser.parse_args()


def canonicalize_smiles(smiles: str) -> str | None:
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def validate_input_frame(df: pd.DataFrame, csv_path: Path, split_name: str) -> pd.DataFrame:
    if "SMILES" not in df.columns:
        raise ValueError(f"{csv_path} is missing the required 'SMILES' column.")

    result = df.copy().reset_index(drop=True)
    if "Molecule Name" not in result.columns:
        result["Molecule Name"] = [f"{split_name}_{idx}" for idx in range(len(result))]
    result["canonical_smiles"] = result["SMILES"].map(canonicalize_smiles)
    invalid_mask = result["canonical_smiles"].isna()
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        invalid_preview = result.loc[invalid_mask, "SMILES"].head(5).tolist()
        raise ValueError(
            f"Found {invalid_count} invalid {split_name} SMILES in {csv_path}. "
            f"Examples: {invalid_preview}"
        )
    return result


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


def greedy_unique_match(
    train_features: np.ndarray,
    test_features: np.ndarray,
) -> pd.DataFrame:
    n_test = test_features.shape[0]
    n_train = train_features.shape[0]
    if n_test > n_train:
        raise ValueError(
            f"Cannot build a one-to-one internal split: {n_test} unique test molecules "
            f"but only {n_train} train rows are available."
        )

    train_features = np.asarray(train_features, dtype=np.float32)
    test_features = np.asarray(test_features, dtype=np.float32)

    test_sq = np.sum(test_features * test_features, axis=1, keepdims=True)
    train_sq = np.sum(train_features * train_features, axis=1, keepdims=True).T
    distance_sq = np.maximum(test_sq + train_sq - 2.0 * (test_features @ train_features.T), 0.0)

    pair_order = np.argsort(distance_sq, axis=None, kind="stable")
    matched_test = np.zeros(n_test, dtype=bool)
    matched_train = np.zeros(n_train, dtype=bool)
    assignments: list[tuple[int, int, float]] = []

    for flat_idx in pair_order:
        test_idx, train_idx = np.unravel_index(flat_idx, distance_sq.shape)
        if matched_test[test_idx] or matched_train[train_idx]:
            continue

        matched_test[test_idx] = True
        matched_train[train_idx] = True
        assignments.append(
            (int(test_idx), int(train_idx), float(np.sqrt(distance_sq[test_idx, train_idx])))
        )
        if len(assignments) == n_test:
            break

    if len(assignments) != n_test:
        raise RuntimeError(
            f"Expected {n_test} assignments but only built {len(assignments)}."
        )

    return (
        pd.DataFrame(assignments, columns=["test_position", "train_position", "distance"])
        .sort_values("test_position")
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()

    train_csv = args.train_csv.resolve()
    test_csv = args.test_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = validate_input_frame(pd.read_csv(train_csv), train_csv, split_name="train")
    test_df = validate_input_frame(pd.read_csv(test_csv), test_csv, split_name="test")

    test_df["test_row_idx"] = np.arange(len(test_df))
    unique_test_df = (
        test_df.drop_duplicates(subset="canonical_smiles", keep="first")
        .copy()
        .reset_index(drop=True)
    )
    unique_test_df["unique_test_position"] = np.arange(len(unique_test_df))

    dropped_duplicate_tests = len(test_df) - len(unique_test_df)
    if dropped_duplicate_tests:
        print(
            f"Removed {dropped_duplicate_tests} duplicate blinded test rows after canonical SMILES deduplication."
        )
    print(
        f"Using {len(unique_test_df)} unique blinded test molecules to build the internal split."
    )

    resolved_device = args.chemeleon_device
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        resolved_device = "cpu"
    print(f"Using CheMeleon device: {resolved_device}")
    fingerprinter = create_chemeleon_fingerprinter(resolved_device)

    train_features = smiles_to_chemeleon_features(
        train_df["SMILES"], fingerprinter, args.batch_size
    )
    unique_test_features = smiles_to_chemeleon_features(
        unique_test_df["SMILES"], fingerprinter, args.batch_size
    )
    print(
        f"Computed CheMeleon features for {len(train_df)} train rows and "
        f"{len(unique_test_df)} unique blinded test rows."
    )

    assignments = greedy_unique_match(train_features=train_features, test_features=unique_test_features)
    selected_train_positions = assignments["train_position"].to_numpy(dtype=np.int64)

    internal_test_df = train_df.iloc[selected_train_positions].copy().reset_index(drop=True)
    internal_test_df["internal_split"] = "test"
    internal_test_df["matched_hidden_test_row_idx"] = unique_test_df["test_row_idx"].to_numpy()
    internal_test_df["matched_hidden_test_molecule_name"] = unique_test_df.get(
        "Molecule Name",
        pd.Series([None] * len(unique_test_df)),
    ).to_numpy()
    internal_test_df["matched_hidden_test_smiles"] = unique_test_df["SMILES"].to_numpy()
    internal_test_df["match_distance"] = assignments["distance"].to_numpy()

    internal_train_df = train_df.drop(index=selected_train_positions).copy().reset_index(drop=True)
    internal_train_df["internal_split"] = "train"

    combined_df = pd.concat([internal_train_df, internal_test_df], ignore_index=True)

    match_df = unique_test_df[
        ["unique_test_position", "test_row_idx", "Molecule Name", "SMILES", "canonical_smiles"]
    ].copy()
    match_df = match_df.rename(
        columns={
            "Molecule Name": "hidden_test_molecule_name",
            "SMILES": "hidden_test_smiles",
            "canonical_smiles": "hidden_test_canonical_smiles",
        }
    )
    match_df["matched_train_position"] = assignments["train_position"].to_numpy()
    match_df["matched_train_molecule_name"] = internal_test_df.get(
        "Molecule Name",
        pd.Series([None] * len(internal_test_df)),
    ).to_numpy()
    match_df["matched_train_smiles"] = internal_test_df["SMILES"].to_numpy()
    match_df["matched_train_canonical_smiles"] = internal_test_df["canonical_smiles"].to_numpy()
    match_df["distance"] = assignments["distance"].to_numpy()

    internal_train_path = output_dir / "internal_train.csv"
    internal_test_path = output_dir / "internal_test.csv"
    combined_path = output_dir / "internal_split.csv"
    matches_path = output_dir / "internal_test_matches.csv"
    summary_path = output_dir / "internal_split_summary.json"

    internal_train_df.to_csv(internal_train_path, index=False)
    internal_test_df.to_csv(internal_test_path, index=False)
    combined_df.to_csv(combined_path, index=False)
    match_df.to_csv(matches_path, index=False)

    summary = {
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "chemeleon_device": resolved_device,
        "batch_size": int(args.batch_size),
        "n_train_old": int(len(train_df)),
        "n_test_old": int(len(test_df)),
        "n_test_unique": int(len(unique_test_df)),
        "n_train_new": int(len(internal_train_df)),
        "n_internal_test": int(len(internal_test_df)),
        "row_identity_holds": bool(len(train_df) == len(internal_train_df) + len(internal_test_df)),
        "unique_test_size_matches_internal_test": bool(
            len(unique_test_df) == len(internal_test_df)
        ),
        "distance_mean": float(assignments["distance"].mean()),
        "distance_median": float(assignments["distance"].median()),
        "distance_max": float(assignments["distance"].max()),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote {internal_train_path}")
    print(f"Wrote {internal_test_path}")
    print(f"Wrote {combined_path}")
    print(f"Wrote {matches_path}")
    print(f"Wrote {summary_path}")
    print(
        "Count check: "
        f"n_train_old={summary['n_train_old']} | "
        f"n_train_new={summary['n_train_new']} | "
        f"n_internal_test={summary['n_internal_test']} | "
        f"identity={summary['row_identity_holds']} | "
        f"unique_test_match={summary['unique_test_size_matches_internal_test']}"
    )


if __name__ == "__main__":
    main()
