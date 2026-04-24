import ctypes
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from rdkit import Chem

from openadmet_pxr_eval import FeatureResult, run_openadmet_pxr_experiment

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

    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
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
    print(f"Loaded SMI-TED model from {model_name} on device {resolved_device}")
    return model, str(resolved_device)


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


if __name__ == "__main__":
    smited_feature_extractor, resolved_device = create_smited_feature_extractor(
        SMITED_MODEL_NAME, SMITED_DEVICE
    )

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
            train_features=smiles_to_smited_features(
                train_df["SMILES"], smited_feature_extractor, SMITED_BATCH_SIZE
            ),
            test_features=smiles_to_smited_features(
                test_df["SMILES"], smited_feature_extractor, SMITED_BATCH_SIZE
            ),
            metadata={
                "smited_model_name": SMITED_MODEL_NAME,
                "smited_variant": SMITED_VARIANT,
                "smited_device": resolved_device,
            },
        )

    run_openadmet_pxr_experiment(
        benchmark_name=BENCHMARK_SET,
        feature_name="smited",
        build_features=build_features,
        feature_config={
            "smited_model_name": SMITED_MODEL_NAME,
            "smited_variant": SMITED_VARIANT,
            "requested_device": SMITED_DEVICE,
            "resolved_device": resolved_device,
            "batch_size": SMITED_BATCH_SIZE,
            "vocab_filename": SMITED_VOCAB_FILENAME,
            "weights_filename": SMITED_WEIGHTS_FILENAME,
            "weights_override_path": SMITED_WEIGHTS_PATH or None,
        },
    )
