import argparse
import ctypes
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/py110287/login23-g-1_34087/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/py110287/login23-g-1_34087/numba_cache")
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from huggingface_hub import hf_hub_download
from rdkit import Chem
from rdkit import RDLogger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

OPENADMET_DIR = Path(__file__).resolve().parent
MOLTABICL_REPO = OPENADMET_DIR.parent
if str(MOLTABICL_REPO) not in sys.path:
    sys.path.insert(0, str(MOLTABICL_REPO))

LOCAL_CLAMP_REPO = OPENADMET_DIR / "clamp"
if LOCAL_CLAMP_REPO.exists() and str(LOCAL_CLAMP_REPO) not in sys.path:
    # Ensure `import clamp` resolves to the local package in `openadmet/clamp/clamp`.
    sys.path.insert(0, str(LOCAL_CLAMP_REPO))

import clamp
from tabpfnmolprop.featurizer import (
    compute_descriptors_from_smiles,
    create_chemeleon_fingerprinter,
    smiles_to_chemeleon_features,
)

warnings.filterwarnings("ignore", category=FutureWarning, module="mordred")
RDLogger.DisableLog("rdApp.*")


def _preload_openmp_runtime() -> None:
    candidate_paths = [
        Path(sys.prefix) / "lib" / "libiomp5.so",
        Path(sys.prefix) / "lib" / "libomp.so",
        Path(sys.prefix) / "lib" / "libgomp.so.1",
    ]
    for lib_path in candidate_paths:
        if lib_path.exists():
            try:
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                return
            except OSError:
                continue


def create_smited_feature_extractor(
    model_name: str,
    device: str,
    variant: str,
    vocab_filename: str,
    weights_filename: str,
    weights_path: str,
):
    if device.startswith("cuda") and torch.cuda.is_available():
        use_cuda = True
        resolved_device = torch.device(device)
    else:
        use_cuda = False
        resolved_device = torch.device("cpu")

    load_path = hf_hub_download(
        repo_id=model_name,
        filename=f"smi-ted/inference/{variant}/load.py",
    )
    vocab_path = hf_hub_download(
        repo_id=model_name,
        filename=f"smi-ted/inference/{variant}/{vocab_filename}",
    )
    config_path = hf_hub_download(repo_id=model_name, filename="config.json")
    selected_weights_path = (
        str(Path(weights_path).resolve())
        if weights_path
        else hf_hub_download(repo_id=model_name, filename=weights_filename)
    )

    module_name = f"smited_{variant}_load"
    spec = importlib.util.spec_from_file_location(module_name, load_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load SMI-TED module spec from {load_path}")
    smited_module = importlib.util.module_from_spec(spec)
    _preload_openmp_runtime()
    try:
        spec.loader.exec_module(smited_module)
    except ModuleNotFoundError as exc:
        if exc.name == "fast_transformers":
            raise RuntimeError(
                "SMI-TED requires `pytorch-fast-transformers`. Install it in this env with:\n"
                f"{sys.executable} -m pip install pytorch-fast-transformers"
            ) from exc
        raise

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config.setdefault("n_output", 1)

    tokenizer = smited_module.MolTranBertTokenizer(str(Path(vocab_path).resolve()))
    model = smited_module.Smi_ted(tokenizer, config=config)
    model_max_len = (
        config.get("max_len")
        or config.get("max_length")
        or config.get("max_seq_len")
        or config.get("max_position_embeddings")
        or getattr(tokenizer, "model_max_length", None)
        or 512
    )
    encoder_n_embd = None
    if hasattr(model, "encoder") and hasattr(model.encoder, "tok_emb"):
        encoder_n_embd = getattr(model.encoder.tok_emb, "embedding_dim", None)
    model_n_embd = (
        config.get("n_embd")
        or config.get("hidden_size")
        or config.get("d_model")
        or encoder_n_embd
    )
    if model_n_embd is None:
        raise RuntimeError("Could not infer SMI-TED embedding size (`n_embd`).")
    model.max_len = int(model_max_len)
    model.n_embd = int(model_n_embd)

    state_dict = torch.load(selected_weights_path, map_location=torch.device("cpu"))
    if not isinstance(state_dict, dict):
        raise RuntimeError("Expected model_weights.bin to contain a state dict mapping.")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"SMI-TED missing keys while loading weights: {len(missing)}")
    if unexpected:
        print(f"SMI-TED unexpected keys while loading weights: {len(unexpected)}")
    print(
        f"Loaded SMI-TED binary weights from {selected_weights_path} "
        f"(vocab size: {len(tokenizer.vocab)})"
    )

    if use_cuda:
        model.cuda()
    else:
        model.cpu()

    for module in [model, getattr(model, "encoder", None), getattr(model, "decoder", None)]:
        if module is None:
            continue
        if hasattr(module, "is_cuda_available"):
            module.is_cuda_available = use_cuda
        for child in module.modules():
            if hasattr(child, "is_cuda_available"):
                child.is_cuda_available = use_cuda

    model.eval()
    print(f"Using SMI-TED device: {resolved_device}")
    return model


def smiles_to_smited_features(
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
        raise RuntimeError("No valid SMILES available for SMI-TED encoding.")

    bs = max(1, int(batch_size))
    with torch.no_grad():
        embeddings = model.encode(valid_smiles, batch_size=bs, return_torch=True)
    if not torch.is_tensor(embeddings):
        embeddings = torch.as_tensor(np.asarray(embeddings), dtype=torch.float32)
    valid_features = embeddings.detach().cpu().numpy().astype(np.float32)

    features = np.zeros((len(smiles_values), valid_features.shape[1]), dtype=np.float32)
    features[np.asarray(valid_positions, dtype=np.int64)] = valid_features
    return features


def create_clamp_encoder(device: str):
    requested = device
    if requested.startswith("cuda") and not torch.cuda.is_available():
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


def load_split_data(train_csv: Path, test_csv: Path) -> pd.DataFrame:
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    if "SMILES" not in train_df.columns or "SMILES" not in test_df.columns:
        raise ValueError("Both train and test CSV files must contain a 'SMILES' column.")

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    test_df["split"] = "test"

    if "Molecule Name" not in train_df.columns:
        train_df["Molecule Name"] = [f"train_{i}" for i in range(len(train_df))]
    if "Molecule Name" not in test_df.columns:
        test_df["Molecule Name"] = [f"test_{i}" for i in range(len(test_df))]

    all_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    all_df["is_valid_smiles"] = all_df["SMILES"].apply(
        lambda s: isinstance(s, str) and Chem.MolFromSmiles(s) is not None
    )
    invalid_count = int((~all_df["is_valid_smiles"]).sum())
    if invalid_count > 0:
        print(f"Found {invalid_count} invalid SMILES. Their embedding rows are kept as zeros.")
    print(
        f"Loaded {len(train_df)} train and {len(test_df)} test molecules ({len(all_df)} total)."
    )
    return all_df


def reduce_with_pca(features: np.ndarray, random_seed: int) -> np.ndarray:
    scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2, random_state=random_seed)
    return pca.fit_transform(scaled).astype(np.float32)


def reduce_with_umap(
    features: np.ndarray,
    random_seed: int,
    n_neighbors: int,
    min_dist: float,
) -> np.ndarray:
    scaled = StandardScaler().fit_transform(features)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_seed,
    )
    return reducer.fit_transform(scaled).astype(np.float32)


def save_scatter_plot(
    coords_2d: np.ndarray,
    split_labels: pd.Series,
    title: str,
    output_path: Path,
) -> None:
    colors = {"train": "#1f77b4", "test": "#ff7f0e"}
    fig, ax = plt.subplots(figsize=(8, 6), dpi=220)
    for split_name in ("train", "test"):
        mask = split_labels.eq(split_name).to_numpy()
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            s=12,
            alpha=0.72,
            c=colors[split_name],
            linewidths=0,
            label=f"{split_name} (n={int(mask.sum())})",
        )

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def save_projection_csv(
    metadata_df: pd.DataFrame,
    coords_2d: np.ndarray,
    output_path: Path,
) -> None:
    projection_df = metadata_df[["Molecule Name", "SMILES", "split", "is_valid_smiles"]].copy()
    projection_df["x"] = coords_2d[:, 0]
    projection_df["y"] = coords_2d[:, 1]
    projection_df.to_csv(output_path, index=False)
    print(f"Saved coordinates: {output_path}")


def run_for_embedding(
    embedding_name: str,
    features: np.ndarray,
    metadata_df: pd.DataFrame,
    output_dir: Path,
    random_seed: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
) -> None:
    features = np.asarray(features, dtype=np.float32)
    nan_count = int(np.isnan(features).sum())
    inf_count = int(np.isinf(features).sum())
    if nan_count or inf_count:
        print(
            f"{embedding_name}: replacing {nan_count} NaN and {inf_count} inf values with 0.0 before PCA/UMAP."
        )
        features = np.nan_to_num(
            features,
            copy=False,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    pca_coords = reduce_with_pca(features, random_seed=random_seed)
    umap_coords = reduce_with_umap(
        features,
        random_seed=random_seed,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
    )

    pca_plot = output_dir / f"{embedding_name.lower()}_pca.png"
    umap_plot = output_dir / f"{embedding_name.lower()}_umap.png"
    pca_csv = output_dir / f"{embedding_name.lower()}_pca_coords.csv"
    umap_csv = output_dir / f"{embedding_name.lower()}_umap_coords.csv"

    save_scatter_plot(
        coords_2d=pca_coords,
        split_labels=metadata_df["split"],
        title=f"{embedding_name} Embedding Projection (PCA)",
        output_path=pca_plot,
    )
    save_scatter_plot(
        coords_2d=umap_coords,
        split_labels=metadata_df["split"],
        title=f"{embedding_name} Embedding Projection (UMAP)",
        output_path=umap_plot,
    )
    save_projection_csv(metadata_df, pca_coords, pca_csv)
    save_projection_csv(metadata_df, umap_coords, umap_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate train-vs-test embedding shift plots (PCA + UMAP) for "
            "CheMeleon, CLAMP, Mordred, and SMI-TED embeddings."
        )
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("openadmet_pxr/openadmetpxr/train.csv"),
        help="Path to training CSV with a SMILES column.",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("openadmet_pxr/openadmetpxr/test.csv"),
        help="Path to test CSV with a SMILES column.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("openadmet_pxr/embedding_analysis"),
        help="Directory where plots and coordinate CSVs are written.",
    )
    parser.add_argument(
        "--chemeleon-device",
        type=str,
        default=os.getenv("CHEMELEON_DEVICE", "cuda"),
        help="Device for CheMeleon fingerprinting.",
    )
    parser.add_argument(
        "--clamp-device",
        type=str,
        default=os.getenv("CLAMP_DEVICE", "cuda"),
        help="Device for CLAMP embeddings.",
    )
    parser.add_argument(
        "--smited-model-name",
        type=str,
        default=os.getenv("SMITED_MODEL_NAME", "ibm-research/materials.smi-ted"),
        help="Hugging Face repo id for SMI-TED.",
    )
    parser.add_argument(
        "--smited-device",
        type=str,
        default=os.getenv("SMITED_DEVICE", "cuda"),
        help="Device for SMI-TED embeddings.",
    )
    parser.add_argument(
        "--smited-batch-size",
        type=int,
        default=int(os.getenv("SMITED_BATCH_SIZE", 64)),
        help="Batch size for SMI-TED encoding.",
    )
    parser.add_argument(
        "--smited-variant",
        type=str,
        default=os.getenv("SMITED_VARIANT", "smi_ted_light"),
        help="SMI-TED inference variant path under the model repository.",
    )
    parser.add_argument(
        "--smited-vocab-filename",
        type=str,
        default=os.getenv("SMITED_VOCAB_FILENAME", "bert_vocab_curated.txt"),
        help="SMI-TED vocab filename under the selected variant directory.",
    )
    parser.add_argument(
        "--smited-weights-filename",
        type=str,
        default=os.getenv("SMITED_WEIGHTS_FILENAME", "model_weights.bin"),
        help="SMI-TED weights filename in the model repository.",
    )
    parser.add_argument(
        "--smited-weights-path",
        type=str,
        default=os.getenv("SMITED_WEIGHTS_PATH", "").strip(),
        help="Optional local path to SMI-TED weights (.bin). If set, skips HF download for weights.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("EMBEDDING_BATCH_SIZE", 256)),
        help="Batch size used for CheMeleon and CLAMP encoders.",
    )
    parser.add_argument(
        "--mordred-nproc",
        type=int,
        default=int(os.getenv("MORDRED_NPROC", 1)),
        help="Number of processes for Mordred descriptor computation.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for PCA/UMAP.",
    )
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=30,
        help="UMAP n_neighbors parameter.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to: {output_dir}")

    data_df = load_split_data(args.train_csv.resolve(), args.test_csv.resolve())
    smiles = data_df["SMILES"]

    resolved_chemeleon_device = args.chemeleon_device
    if resolved_chemeleon_device.startswith("cuda") and not torch.cuda.is_available():
        resolved_chemeleon_device = "cpu"
    print(f"Using CheMeleon device: {resolved_chemeleon_device}")
    chemeleon_fingerprinter = create_chemeleon_fingerprinter(resolved_chemeleon_device)
    chemeleon_features = smiles_to_chemeleon_features(
        smiles,
        chemeleon_fingerprinter,
        args.batch_size,
    ).astype(np.float32)
    print(f"CheMeleon features: {chemeleon_features.shape}")

    clamp_encoder = create_clamp_encoder(args.clamp_device)
    clamp_features = smiles_to_clamp_features(
        smiles,
        clamp_encoder,
        args.batch_size,
    ).astype(np.float32)
    print(f"CLAMP features: {clamp_features.shape}")

    mordred_features = compute_descriptors_from_smiles(
        smiles,
        nproc=args.mordred_nproc,
    ).astype(np.float32)
    print(f"Mordred features: {mordred_features.shape}")

    smited_feature_extractor = create_smited_feature_extractor(
        model_name=args.smited_model_name,
        device=args.smited_device,
        variant=args.smited_variant,
        vocab_filename=args.smited_vocab_filename,
        weights_filename=args.smited_weights_filename,
        weights_path=args.smited_weights_path,
    )
    smited_features = smiles_to_smited_features(
        smiles,
        smited_feature_extractor,
        args.smited_batch_size,
    ).astype(np.float32)
    print(f"SMI-TED features: {smited_features.shape}")

    run_for_embedding(
        embedding_name="CheMeleon",
        features=chemeleon_features,
        metadata_df=data_df,
        output_dir=output_dir,
        random_seed=args.random_seed,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
    )
    run_for_embedding(
        embedding_name="CLAMP",
        features=clamp_features,
        metadata_df=data_df,
        output_dir=output_dir,
        random_seed=args.random_seed,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
    )
    run_for_embedding(
        embedding_name="Mordred",
        features=mordred_features,
        metadata_df=data_df,
        output_dir=output_dir,
        random_seed=args.random_seed,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
    )
    run_for_embedding(
        embedding_name="SMITED",
        features=smited_features,
        metadata_df=data_df,
        output_dir=output_dir,
        random_seed=args.random_seed,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
    )
    print("Done.")


if __name__ == "__main__":
    main()
