"""
Minimal R wrapper runner (CSV handoff).

Policy:
- Python owns plotting.
- R is invoked only for book-faithful computations that are not practical to reproduce.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


CORE_DIR = Path(__file__).resolve().parent
PYEXPDES_DIR = CORE_DIR.parent
R_WRAPPERS_DIR = PYEXPDES_DIR / "r_wrappers"


class RUnavailable(RuntimeError):
    pass


def _rscript() -> str:
    r = shutil.which("Rscript")
    if not r:
        raise RUnavailable("Rscript not found on PATH.")
    return r


def ensure_r_packages(packages: Iterable[str]) -> None:
    pkgs = list(packages)
    if not pkgs:
        return
    code = (
    "pkgs <- c("
    + ",".join([f"'{p}'" for p in pkgs])
    + "); missing <- pkgs[!sapply(pkgs, requireNamespace, quietly=TRUE)];"
    + " if(length(missing)>0){ stop(paste('Missing R packages:', paste(missing, collapse=', '))) }"
    )

    subprocess.run([_rscript(), "-e", code], check=True, capture_output=True, text=True)


def _get_wrapper_path(wrapper_name: str) -> Path:
    import re
    match = re.match(r"ch(\d+)", wrapper_name)
    if match:
        ch_num = int(match.group(1))
        chapter_dir = f"ch{ch_num:02d}"
        return R_WRAPPERS_DIR / chapter_dir / wrapper_name
    return R_WRAPPERS_DIR / wrapper_name


def run_wrapper(
    wrapper_name: str,
    *,
    inputs: dict[str, pd.DataFrame],
    args: list[str],
    required_packages: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    wrapper = _get_wrapper_path(wrapper_name)
    if not wrapper.exists():
        raise FileNotFoundError(f"R wrapper not found: {wrapper}")

    if required_packages:
        ensure_r_packages(required_packages)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_dir = td / "in"
        out_dir = td / "out"
        in_dir.mkdir()
        out_dir.mkdir()

        for name, df in inputs.items():
            df.to_csv(in_dir / name, index=False)

        cmd = [_rscript(), str(wrapper), str(in_dir), str(out_dir), *args]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "R wrapper failed.\n"
                f"cmd: {cmd}\n\n"
                f"stdout:\n{exc.stdout}\n\n"
                f"stderr:\n{exc.stderr}"
            ) from exc

        outputs: dict[str, pd.DataFrame] = {}
        for p in out_dir.glob("*.csv"):
            outputs[p.name] = pd.read_csv(p)
        return outputs


