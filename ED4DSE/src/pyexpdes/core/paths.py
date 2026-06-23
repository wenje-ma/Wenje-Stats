from __future__ import annotations

from pathlib import Path


CORE_DIR = Path(__file__).resolve().parent
PYEXPDES_DIR = CORE_DIR.parent
SRC_DIR = PYEXPDES_DIR.parent
REPO_ROOT = SRC_DIR.parent


def figures_dir(module: str) -> Path:
    d = REPO_ROOT / "figures" / module
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_wrapper_path(chapter: str, script_name: str) -> Path:
    return PYEXPDES_DIR / "r_wrappers" / chapter / script_name


def datasets_dir() -> Path:
    return REPO_ROOT / "datasets"


def notebooks_dir() -> Path:
    return REPO_ROOT / "notebooks"


