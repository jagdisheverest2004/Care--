import argparse
import sys
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from kaggle.api.kaggle_api_extended import KaggleApi

from config import ensure_dirs, DATASETS_DIR
from project_datasets import DATASETS, get_dataset_dir


NON_KAGGLE = {
    "drugbank": "https://go.drugbank.com/releases/latest",
}


def _download_kaggle_dataset(api: KaggleApi, slug: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(slug, path=str(out_dir), unzip=True, quiet=False)


def download(keys: List[str]) -> None:
    ensure_dirs()
    api = KaggleApi()
    api.authenticate()

    for key in keys:
        info = DATASETS.get(key)
        if not info:
            print(f"Skipping unknown dataset key: {key}")
            continue
        out_dir = get_dataset_dir(key)
        print(f"Downloading {key} -> {out_dir}")
        _download_kaggle_dataset(api, info["slug"], out_dir)

    if NON_KAGGLE:
        print("\nNon-Kaggle datasets require manual download:")
        for name, url in NON_KAGGLE.items():
            print(f"- {name}: {url}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Download all Kaggle datasets")
    parser.add_argument("--datasets", nargs="*", default=[], help="Dataset keys to download")
    args = parser.parse_args()

    if args.all:
        keys = list(DATASETS.keys())
    else:
        keys = args.datasets

    if not keys:
        print("No datasets selected. Use --all or --datasets <keys>.")
        print("Available keys:")
        for key in DATASETS:
            print(f"- {key}")
        return

    download(keys)


if __name__ == "__main__":
    main()
