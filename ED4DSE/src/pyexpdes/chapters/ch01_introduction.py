"""
Python port of `R files/Ch1.R` (Introduction).

This module generates two figures:
1) `design_objectives.png`  (4 panels, axes hidden)
2) `regression_comparison.png` (linear regression vs Gaussian process regression)

Python-first policy:
- Plotting is done in Python.
- R is invoked only where the book uses specialized packages:
  - `support::sp` for the space-filling design (uncertainty propagation panel)
  - `rkriging` for Gaussian process regression (right panel)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta as sp_beta
from scipy.stats import t as student_t

from pyexpdes.core.paths import figures_dir
from pyexpdes.core.style import apply_base_style
from pyexpdes.core.r_utils import run_wrapper


def _f_objectives_rough(x: np.ndarray) -> np.ndarray:
    # introduction.R uses (x - 0.5) for the objectives figure
    return np.sin(10 * np.pi * x) / (1 + 64 * (x - 0.5) ** 2)


def _f_objectives_smooth(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1 + 64 * (x - 0.5) ** 2)


def _f_mixture(x: np.ndarray) -> np.ndarray:
    return (2.0 / 3.0) * sp_beta.pdf(x, 2, 10) + (1.0 / 3.0) * sp_beta.pdf(x, 10, 2)


def _f_regression(x: np.ndarray) -> np.ndarray:
    # regression section uses (x - 0.25)
    return np.sin(10 * np.pi * x) / (1 + 64 * (x - 0.25) ** 2) + x**2


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _prediction_interval_lm(
    x: np.ndarray, y: np.ndarray, x_new: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match R's `predict.lm(..., interval='prediction')` (default 95%).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x_new = np.asarray(x_new, float)

    X = np.column_stack([np.ones_like(x), x])
    XtX = X.T @ X
    beta_hat = np.linalg.solve(XtX, X.T @ y)

    y_hat = X @ beta_hat
    resid = y - y_hat

    n = len(x)
    p = X.shape[1]
    dof = n - p
    s2 = float((resid @ resid) / dof)

    Xn = np.column_stack([np.ones_like(x_new), x_new])
    XtX_inv = np.linalg.inv(XtX)
    h = np.einsum("ij,jk,ik->i", Xn, XtX_inv, Xn)

    se_pred = np.sqrt(s2 * (1.0 + h))
    tcrit = float(student_t.ppf(0.975, dof))

    mean = Xn @ beta_hat
    lo = mean - tcrit * se_pred
    hi = mean + tcrit * se_pred
    return mean, lo, hi


def generate_figures(*, use_r: bool = True) -> tuple[Path, list[Path]]:
    """
    Generate introduction figures.

    Returns (output_dir, [figure_paths...]).
    """
    apply_base_style()
    out_dir = figures_dir("ch01")
    paths: list[Path] = []

    # ------------------------------------------------------------------
    # Figure: design objectives (2x2, axes=FALSE)
    # ------------------------------------------------------------------
    x = np.linspace(0.0, 1.0, 301)
    n = 15

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    # Panel 1: rough
    y = _f_objectives_rough(x)
    ax = axes[0]
    ax.plot(x, y, color="black", linewidth=3)
    ax.set_title("approximate a rough function", fontsize=18)
    D = (np.arange(1, n + 1) - 0.5) / n
    ax.scatter(D, np.full_like(D, y.min() - 0.05), s=60, color="blue")
    ax.set_axis_off()

    # Panel 2: smooth
    y = _f_objectives_smooth(x)
    ax = axes[1]
    ax.plot(x, y, color="black", linewidth=3)
    ax.set_title("approximate a smooth function", fontsize=18)
    D = sp_beta.ppf((np.arange(1, n + 1) - 0.5) / n, 0.5, 0.5)
    ax.scatter(D, np.full_like(D, y.min() - 0.05), s=60, color="blue")
    ax.set_axis_off()

    # Panel 3: optimization
    y = _f_mixture(x)
    ax = axes[2]
    ax.plot(x, y, color="black", linewidth=3)
    ax.set_title("optimization", fontsize=18)
    n1 = int(n * 2 / 3)
    n2 = n - n1
    D0 = (np.arange(1, n2 + 1) - 0.5) / n2
    D = np.concatenate([D0, sp_beta.ppf((np.arange(1, n1 + 1) - 0.5) / n1, 2, 10)])
    ax.scatter(D, np.full_like(D, y.min() - 0.05), s=60, color="blue")
    ax.set_axis_off()

    # Panel 4: uncertainty propagation (support::sp)
    y = _f_mixture(x)
    ax = axes[3]
    ax.plot(x, y, color="black", linewidth=3)
    ax.set_title("uncertainty propagation", fontsize=18)
    ax.set_axis_off()

    rng = np.random.default_rng(0)
    dist_samp = np.concatenate(
        [rng.beta(2, 10, 1000 * n1), rng.beta(10, 2, 1000 * n2)]
    )
    if use_r:
        out = run_wrapper(
            "ch1_support_sp.R",
            inputs={"dist_samples.csv": pd.DataFrame({"x": dist_samp})},
            args=[str(n)],
            required_packages=["support"],
        )
        D = out["design.csv"]["D"].to_numpy()
    else:
        D = np.quantile(dist_samp, (np.arange(1, n + 1) - 0.5) / n)

    ax.scatter(D, np.full_like(D, y.min() - 0.05), s=60, color="blue")

    fig.tight_layout()
    paths.append(_save(fig, out_dir / "design_objectives.png"))

    # ------------------------------------------------------------------
    # Figure: linear regression vs GP (1x2)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(1)  # set.seed(1)
    n = 10
    x_obs = np.array([0.0] * (n // 2) + [1.0] * (n // 2))
    y_obs = _f_regression(x_obs) + 0.1 * rng.normal(size=n)

    l, u = -0.1, 1.1
    test = np.arange(l, u + 0.0001, 0.001)
    truey = _f_regression(test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear regression (prediction interval)
    ax = axes[0]
    ax.plot(test, truey, color="black", linewidth=2)
    ax.set_xlim(l, u)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title("Linear regression", fontsize=16)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)

    mean, lo, hi = _prediction_interval_lm(x_obs, y_obs, test)
    ax.plot(test, mean, color="green", linewidth=3, linestyle="--")
    ax.fill_between(test, lo, hi, color="pink", alpha=1.0, linewidth=0)
    ax.plot(test, lo, color="pink", linewidth=2)
    ax.plot(test, hi, color="pink", linewidth=2)
    ax.scatter(x_obs, y_obs, s=60, color="blue")
    ax.legend(["truth", "prediction"], loc="upper left", frameon=False, fontsize=12)

    # Gaussian process regression (rkriging)
    ax = axes[1]
    ax.plot(test, truey, color="black", linewidth=2)
    ax.set_xlim(l, u)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title("Gaussian process regression", fontsize=16)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)

    if use_r:
        data = pd.concat(
            [
                pd.DataFrame({"x": x_obs, "y": y_obs, "test": np.nan}),
                pd.DataFrame({"x": np.nan, "y": np.nan, "test": test}),
            ],
            ignore_index=True,
        )
        out = run_wrapper(
            "ch1_ch2_rkriging_predict.R",
            inputs={"data.csv": data},
            args=[],
            required_packages=["rkriging"],
        )
        pred = out["pred.csv"]
        gp_mean = pred["mean"].to_numpy()
        gp_sd = pred["sd"].to_numpy()
    else:
        # Non-book fallback
        gp_mean = mean
        gp_sd = np.full_like(gp_mean, 0.1)

    ax.plot(test, gp_mean, color="green", linewidth=3, linestyle="--")
    ax.fill_between(test, gp_mean - 2 * gp_sd, gp_mean + 2 * gp_sd, color="pink", alpha=1.0, linewidth=0)
    ax.plot(test, gp_mean - 2 * gp_sd, color="pink", linewidth=2)
    ax.plot(test, gp_mean + 2 * gp_sd, color="pink", linewidth=2)
    ax.scatter(x_obs, y_obs, s=60, color="blue")
    ax.legend(["truth", "prediction"], loc="upper left", frameon=False, fontsize=12)

    fig.tight_layout()
    paths.append(_save(fig, out_dir / "regression_comparison.png"))

    return out_dir, paths


def main(argv: Sequence[str] | None = None) -> int:
    generate_figures(use_r=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
