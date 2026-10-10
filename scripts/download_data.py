"""Download the pork-rasher packaging dataset from Roboflow.

The Roboflow API key is read from the ``ROBOFLOW_API_KEY`` environment
variable (or from a local ``.env`` file, which is git-ignored). The key must
never be committed to the repository.

Usage (from the repository root)::

    python scripts/download_data.py

The dataset is written to ``data/pork-rasher-error-packaging-<version>/``.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

WORKSPACE = "hello-2aqe0"
PROJECT = "pork-rasher-error-packaging"
VERSION = 4
EXPORT_FORMAT = "yolov12"

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def main() -> None:
    """Download the configured dataset version into ``data/``."""
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit(
            "ROBOFLOW_API_KEY is not set. Copy .env.example to .env and add "
            "your key, or set the environment variable."
        )

    location = DATA_DIR / f"{PROJECT}-{VERSION}"
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    project.version(VERSION).download(EXPORT_FORMAT, location=str(location))
    print(f"Dataset downloaded to {location}")


if __name__ == "__main__":
    main()
