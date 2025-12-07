import argparse
import os
from pathlib import Path

from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
from huggingface_hub import snapshot_download
from tqdm import tqdm

# Explicit dataset URLs (BEIR expects full URLs, not just names)
DATASET_URLS = {
    "fiqa": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
}
DATASETS = list(DATASET_URLS.keys())

EMBED_MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-base": "BAAI/bge-base-en",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}


def prep_datasets(target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        ds_dir = target_dir / ds
        if ds_dir.exists():
            print(f"[datasets] {ds} already present, skipping")
            continue
        print(f"[datasets] downloading {ds} -> {ds_dir}")
        url = DATASET_URLS[ds]
        url_path = beir_util.download_and_unzip(url, str(target_dir))
        # Ensure readable by BEIR loader (validates structure)
        GenericDataLoader(str(url_path)).load(split="test")
        print(f"[datasets] done {ds}")


def prep_models(target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    for short, repo in EMBED_MODELS.items():
        model_dir = target_dir / short
        if model_dir.exists():
            print(f"[models] {short} already cached at {model_dir}, skipping")
            continue
        print(f"[models] caching {short} ({repo}) -> {model_dir}")
        snapshot_download(repo_id=repo, local_dir=str(model_dir), local_dir_use_symlinks=False)
        print(f"[models] done {short}")


def main():
    parser = argparse.ArgumentParser(description="Pre-download BEIR datasets and embedding models")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Target directory for datasets")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Target directory for models")
    args = parser.parse_args()

    prep_datasets(args.data_dir)
    prep_models(args.model_dir)
    print("Preparation complete.")


if __name__ == "__main__":
    main()

