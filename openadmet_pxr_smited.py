import ctypes
import importlib.util
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
from huggingface_hub import hf_hub_download
from rdkit import Chem
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

BENCHMARK_SET = os.getenv("BENCHMARK_SET", "openadmetpxr_smited")
SMITED_MODEL_NAME = os.getenv("SMITED_MODEL_NAME", "ibm-research/materials.smi-ted")
SMITED_DEVICE = os.getenv("SMITED_DEVICE", "cuda")
SMITED_BATCH_SIZE = int(os.getenv("SMITED_BATCH_SIZE", 64))
SMITED_VARIANT = os.getenv("SMITED_VARIANT", "smi_ted_light")
SMITED_VOCAB_FILENAME = os.getenv("SMITED_VOCAB_FILENAME", "bert_vocab_curated.txt")
SMITED_WEIGHTS_FILENAME = os.getenv("SMITED_WEIGHTS_FILENAME", "model_weights.bin")
SMITED_WEIGHTS_PATH = os.getenv("SMITED_WEIGHTS_PATH", "").strip()
CV_N_SPLITS = int(os.getenv("CV_N_SPLITS", 5))
CV_RANDOM_SEED = int(os.getenv("CV_RANDOM_SEED", 42))
print(f"Running benchmark set {BENCHMARK_SET}")


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


def create_smited_feature_extractor(model_name: str, device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        use_cuda = True
        resolved_device = torch.device(device)
    else:
        use_cuda = False
        resolved_device = torch.device("cpu")

    load_path = hf_hub_download(
        repo_id=model_name,
        filename=f"smi-ted/inference/{SMITED_VARIANT}/load.py",
    )
    vocab_path = hf_hub_download(
        repo_id=model_name,
        filename=f"smi-ted/inference/{SMITED_VARIANT}/{SMITED_VOCAB_FILENAME}",
    )
    config_path = hf_hub_download(repo_id=model_name, filename="config.json")
    weights_path = (
        str(Path(SMITED_WEIGHTS_PATH).resolve())
        if SMITED_WEIGHTS_PATH
        else hf_hub_download(repo_id=model_name, filename=SMITED_WEIGHTS_FILENAME)
    )

    module_name = f"smited_{SMITED_VARIANT}_load"
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

    with open(config_path, "r") as f:
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
    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    if not isinstance(state_dict, dict):
        raise RuntimeError("Expected model_weights.bin to contain a state dict mapping.")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"SMI-TED missing keys while loading weights: {len(missing)}")
    if unexpected:
        print(f"SMI-TED unexpected keys while loading weights: {len(unexpected)}")

    print(
        f"Loaded SMI-TED binary weights from {weights_path} "
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
    smited_feature_extractor = create_smited_feature_extractor(
        SMITED_MODEL_NAME, SMITED_DEVICE
    )
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

        #save the train and test data for reference
        train_df.to_csv(output_dir / "train.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)

        train_smiles = train_df["SMILES"]
        test_smiles = test_df["SMILES"]
        raw_targets = train_df["pEC50"].to_numpy().ravel()

        # Compute frozen SMI-TED embeddings from SMILES.
        train_features = smiles_to_smited_features(
            train_smiles, smited_feature_extractor, SMITED_BATCH_SIZE
        )
        test_features = smiles_to_smited_features(
            test_smiles, smited_feature_extractor, SMITED_BATCH_SIZE
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

        # Convert to torch tensors for use with standard_scale
        train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
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
        test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
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
