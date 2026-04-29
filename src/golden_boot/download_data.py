from __future__ import annotations

from pathlib import Path

from .config import KAGGLE_DATASETS, RAW_DIR


def download_all(force: bool = False) -> None:
    """Download all Kaggle datasets used by the pipeline."""
    import kagglehub

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for name, spec in KAGGLE_DATASETS.items():
        output_dir = RAW_DIR / spec["folder"]
        already_downloaded = output_dir.exists() and any(output_dir.iterdir())

        if already_downloaded and not force:
            print(f"Skipping {name}: found {output_dir}")
            continue

        print(f"Downloading {name}: {spec['handle']}")
        path = kagglehub.dataset_download(
            spec["handle"],
            output_dir=str(output_dir),
            force_download=force,
        )
        print(f"  saved to {Path(path)}")
