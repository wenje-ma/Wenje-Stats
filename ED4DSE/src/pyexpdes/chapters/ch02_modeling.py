"""
Python port of `R files/Ch2.R` (Modeling).

Policy:
- Plotting is done in Python.
- R is used only for book-faithful steps that rely on specific R packages:
  - `rkriging` for Gaussian process panels (when `use_r=True`)
  - `AlgDesign::optFederov` for D-optimal points (when `use_r=True`)

This implementation focuses on the core figures present in the R script:
- Polynomial interpolation (equi-spaced vs Chebyshev)
- 2D design examples
- RBF interpolation (theta = 0.01, 0.1, 1.0)
- Ordinary kriging
- Gaussian process (via rkriging wrapper when enabled)
- Polynomial regression (equi-spaced vs D-optimal)
- Kernel ridge regression (equi-spaced vs D-optimal)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.optimize import minimize

from pyexpdes.core.paths import figures_dir
from pyexpdes.core.style import apply_base_style
from pyexpdes.core.r_utils import run_wrapper


def _f(x: np.ndarray) -> np.ndarray:
    return np.sin(10 * np.pi * x) / (1 + 64 * (x - 0.25) ** 2) + x**2


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _chebyshev_interp_coeffs(D: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Match the R script's Chebyshev polynomial interpolation:
    X[i,j] = cos((j-1) * acos(d_i)), where d_i = 2*D_i - 1.
    """
    d = 2 * D - 1
    n = len(D)
    X = np.column_stack([np.cos((j) * np.arccos(d)) for j in range(n)])
    return np.linalg.solve(X, y)


def _chebyshev_eval(a: np.ndarray, u: np.ndarray) -> np.ndarray:
    us = 2 * u - 1
    n = len(a)
    T = np.column_stack([np.cos(j * np.arccos(us)) for j in range(n)])
    return T @ a


def _rbf_interp(D: np.ndarray, y: np.ndarray, theta: float, test: np.ndarray) -> np.ndarray:
    E = np.abs(D[:, None] - D[None, :])
    R = np.exp(-((E / theta) ** 2))
    coef = np.linalg.solve(R, y)
    M = np.exp(-(((test[:, None] - D[None, :]) / theta) ** 2))
    return M @ coef


def _ordinary_kriging(D: np.ndarray, y: np.ndarray, theta: float, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Match the R script's ordinary kriging implementation (Gaussian kernel).
    Returns (mean, sd) on `test`.
    """
    n = len(D)
    E = np.abs(D[:, None] - D[None, :])
    R = np.exp(-((E / theta) ** 2))
    one = np.ones(n)
    Rinv = np.linalg.inv(R + 1e-10 * np.eye(n))
    mu = float((one @ Rinv @ y) / (one @ Rinv @ one))
    coef = Rinv @ (y - mu)
    tau2 = float((y - mu).T @ Rinv @ (y - mu) / n)

    M = np.exp(-(((test[:, None] - D[None, :]) / theta) ** 2))
    mean = mu + M @ coef

    fac = 1.0 / float(one @ Rinv @ one)
    # variance expression from the R script
    v = []
    for i in range(len(test)):
        r = M[i, :]
        term = 1.0 - (r @ Rinv @ r) + fac * (1.0 - (r @ Rinv @ one)) ** 2
        v.append(tau2 * term)
    sd = np.sqrt(np.maximum(v, 0.0))
    return mean, sd


def _opt_federov_points(n: int, use_r: bool) -> np.ndarray:
    """
    Return D-optimal points in [0,1] corresponding to optFederov on x in [-1,1].
    """
    if not use_r:
        # Fallback: use equi-spaced (not book-faithful)
        return (np.arange(1, n + 1) - 1) / (n - 1)

    out = run_wrapper(
        "ch2_optfederov.R",
        inputs={"noop.csv": pd.DataFrame({"x": [0.0]})},
        args=[str(n)],
        required_packages=["AlgDesign"],
    )
    x = out["design.csv"]["x"].to_numpy()
    return (x + 1.0) / 2.0


def generate_figures(*, use_r: bool = True) -> tuple[Path, list[Path]]:
    apply_base_style()
    out_dir = figures_dir("ch02")
    paths: list[Path] = []

    test = np.linspace(0, 1, 301)
    true = _f(test)

    # ------------------------------------------------------------------
    # Polynomial interpolation (1x2)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n = 10

    # Equi-spaced
    D = (np.arange(1, n + 1) - 1) / (n - 1)
    y = _f(D)
    a = _chebyshev_interp_coeffs(D, y)
    pred = _chebyshev_eval(a, test)
    ax = axes[0]
    ax.plot(test, true, color="black", linewidth=3, linestyle="--")
    ax.scatter(D, y, s=80, color="blue")
    ax.plot(test, pred, color="green", linewidth=3)
    ax.set_ylim(true.min() - 0.25, true.max() + 0.25)
    ax.set_title("polynomial interpolation equi-spaced points", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(["truth", "prediction"], loc="lower right", frameon=False)

    # Chebyshev nodes
    d = np.cos((2 * (np.arange(1, n + 1)) - 1) / (2 * n) * np.pi)
    D = (d + 1) / 2
    y = _f(D)
    # In this case, the R script uses d directly in X; our helper does the right thing.
    a = _chebyshev_interp_coeffs(D, y)
    pred = _chebyshev_eval(a, test)
    ax = axes[1]
    ax.plot(test, true, color="black", linewidth=3, linestyle="--")
    ax.scatter(D, y, s=80, color="blue")
    ax.plot(test, pred, color="green", linewidth=3)
    ax.set_ylim(true.min() - 0.25, true.max() + 0.25)
    ax.set_title("polynomial interpolation with Chebyshev nodes", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(["truth", "prediction"], loc="lower right", frameon=False)

    fig.tight_layout()
    paths.append(_save(fig, out_dir / "polynomial_interpolation.png"))

    # ------------------------------------------------------------------
    # Two-dimensional examples (1x2)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    D1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    D2 = np.array([[0, 0], [0.2, 0.9], [0.9, 0.1], [1, 1]], float)
    for ax, D in zip(axes, [D1, D2]):
        ax.scatter(D[:, 0], D[:, 1], s=200, color="blue")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "two_dimensional.png"))

    # ------------------------------------------------------------------
    # RBF interpolation (1x3)
    # ------------------------------------------------------------------
    n = 10
    D = (np.arange(1, n + 1) - 1) / (n - 1)
    y = _f(D)
    thetas = [0.01, 0.1, 1.0]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, th in zip(axes, thetas):
        pred = _rbf_interp(D, y, th, test)
        ax.plot(test, true, color="black", linewidth=3, linestyle="--")
        ax.scatter(D, y, s=80, color="blue")
        ax.plot(test, pred, color="green", linewidth=3)
        ax.set_ylim(true.min() - 0.25, true.max() + 0.25)
        ax.set_title(rf"$\theta={th}$", fontsize=14)
        ax.legend(["truth", "prediction"], loc="lower right", frameon=False)
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "rbf_interpolation.png"))

    # ------------------------------------------------------------------
    # Ordinary kriging (single panel)
    # ------------------------------------------------------------------
    n = 10
    D = (np.arange(1, n + 1) - 1) / (n - 1)
    y = _f(D)
    theta = 0.08
    mean, sd = _ordinary_kriging(D, y, theta, test)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(test, true, color="black", linewidth=3, linestyle="--")
    ax.scatter(D, y, s=80, color="blue")
    ax.fill_between(test, mean - 2 * sd, mean + 2 * sd, color="pink", alpha=1.0, linewidth=0)
    ax.plot(test, mean, color="green", linewidth=3, linestyle="--")
    ax.set_ylim(true.min() - 0.25, true.max() + 0.25)
    ax.set_title("ordinary kriging", fontsize=14)
    ax.legend(["truth", "prediction"], loc="lower right", frameon=False)
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "ordinary_kriging.png"))

    # ------------------------------------------------------------------
    # Gaussian process 3-panel: prior, data, posterior (book-faithful via rkriging)
    # ------------------------------------------------------------------
    if use_r:
        n = 10
        D = (np.arange(1, n + 1) - 1) / (n - 1)
        y = _f(D)
        theta = 0.08

        data = pd.concat(
            [
                pd.DataFrame({"x": D, "y": y, "test": np.nan}),
                pd.DataFrame({"x": np.nan, "y": np.nan, "test": test}),
            ],
            ignore_index=True,
        )
        out = run_wrapper(
            "ch2_gp_visualization.R",
            inputs={
                "data.csv": data,
                "params.csv": pd.DataFrame({"theta": [theta], "n_samples": [5], "seed": [1]}),
            },
            args=[],
            required_packages=["rkriging", "MASS", "pdist", "Matrix"],
        )
        prior_samples = out["prior_samples.csv"].values
        posterior_samples = out["posterior_samples.csv"].values
        pred_gp = out["prediction.csv"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Left: Prior distribution
        ax = axes[0]
        for i in range(prior_samples.shape[1]):
            ax.plot(test, prior_samples[:, i], linewidth=2)
        ax.set_ylim(true.min() - 0.25, true.max() + 0.25)
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.set_title("prior distribution", fontsize=14)

        # Middle: Data
        ax = axes[1]
        for i in range(prior_samples.shape[1]):
            ax.plot(test, prior_samples[:, i], linewidth=2)
        ax.scatter(D, y, s=120, color="blue", zorder=5)
        ax.set_ylim(true.min() - 0.25, true.max() + 0.25)
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.set_title("data", fontsize=14)

        # Right: Posterior distribution
        ax = axes[2]
        for i in range(posterior_samples.shape[1]):
            ax.plot(test, posterior_samples[:, i], linewidth=2)
        ax.scatter(D, y, s=120, color="blue", zorder=5)
        ax.set_ylim(true.min() - 0.25, true.max() + 0.25)
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.set_title("posterior distribution", fontsize=14)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "gaussian_process_3panel.png"))

    # ------------------------------------------------------------------
    # Gaussian process single panel (book-faithful via rkriging when enabled)
    # ------------------------------------------------------------------
    if use_r:
        n = 10
        D = (np.arange(1, n + 1) - 1) / (n - 1)
        y = _f(D)
        data = pd.concat(
            [
                pd.DataFrame({"x": D, "y": y, "test": np.nan}),
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
        mean = pred["mean"].to_numpy()
        sd = pred["sd"].to_numpy()
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot(test, true, color="black", linewidth=3, linestyle="--")
        ax.scatter(D, y, s=80, color="blue")
        ax.fill_between(test, mean - 2 * sd, mean + 2 * sd, color="pink", alpha=1.0, linewidth=0)
        ax.plot(test, mean, color="green", linewidth=3, linestyle="--")
        ax.set_ylim(true.min() - 0.25, true.max() + 0.25)
        ax.set_title("Gaussian process", fontsize=14)
        ax.legend(["truth", "prediction"], loc="lower right", frameon=False)
        fig.tight_layout()
        paths.append(_save(fig, out_dir / "gaussian_process.png"))

    # ------------------------------------------------------------------
    # Polynomial regression 3rd degree (equi-spaced vs D-optimal)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(5)  # set.seed(5)
    n = 10
    r = 2
    e = rng.normal(scale=0.1, size=n * r)
    D_equi = np.repeat((np.arange(1, n + 1) - 1) / (n - 1), r)
    y = _f(D_equi) + e

    D_opt = np.repeat(_opt_federov_points(n, use_r=use_r), r)
    y_opt = _f(D_opt) + e

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (title, D, yy) in zip(
        axes,
        [
            ("3rd degree polynomial regression (equi-spaced)", D_equi, y),
            ("3rd degree polynomial regression (D-optimal)", D_opt, y_opt),
        ],
    ):
        ax.plot(test, true, color="black", linewidth=3, linestyle="--")
        ax.scatter(D, yy, s=40, color="blue")
        d = 2 * D - 1
        X = np.vstack([d**k for k in range(0, 4)]).T
        coef = np.linalg.lstsq(X, yy, rcond=None)[0]
        us = 2 * test - 1
        Xs = np.vstack([us**k for k in range(0, 4)]).T
        pred = Xs @ coef
        ax.plot(test, pred, color="green", linewidth=3)
        ax.set_ylim(true.min() - 0.25, true.max() + 0.25)
        ax.set_title(title, fontsize=12)
        ax.legend(["truth", "prediction"], loc="lower right", frameon=False)
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "polynomial_regression_3deg.png"))

    # ------------------------------------------------------------------
    # Polynomial regression 8th degree (equi-spaced vs D-optimal)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (title, D, yy) in zip(
        axes,
        [
            ("8th degree polynomial regression (equi-spaced)", D_equi, y),
            ("8th degree polynomial regression (D-optimal)", D_opt, y_opt),
        ],
    ):
        ax.plot(test, true, color="black", linewidth=3, linestyle="--")
        ax.scatter(D, yy, s=40, color="blue")
        d = 2 * D - 1
        X = np.vstack([d**k for k in range(0, 9)]).T
        coef = np.linalg.lstsq(X, yy, rcond=None)[0]
        us = 2 * test - 1
        Xs = np.vstack([us**k for k in range(0, 9)]).T
        pred = Xs @ coef
        ax.plot(test, pred, color="green", linewidth=3)
        ax.set_ylim(true.min() - 0.25, true.max() + 0.25)
        ax.set_title(title, fontsize=12)
        ax.legend(["truth", "prediction"], loc="lower right", frameon=False)
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "polynomial_regression_8deg.png"))

    # ------------------------------------------------------------------
    # Kernel ridge regression (equi-spaced vs D-optimal) - Pure Python
    # ------------------------------------------------------------------
    def _kernel_ridge_cv(D: np.ndarray, y: np.ndarray, test: np.ndarray) -> np.ndarray:
        """Kernel ridge regression with MSCV hyperparameter tuning."""
        E = np.abs(D[:, None] - D[None, :])
        n_obs = len(D)

        def mscv(para):
            theta, lam = para[0], para[1]
            R = np.exp(-((E / theta) ** 2))
            Rinv = np.linalg.solve(R + lam * np.eye(n_obs), np.eye(n_obs))
            cv = (Rinv @ y) / np.diag(Rinv)
            return np.log(np.mean(cv ** 2))

        result = minimize(mscv, x0=[0.1, 0.1], bounds=[(0.05, 1.0), (0.001, 1.0)], method="L-BFGS-B")
        theta, lam = result.x

        R = np.exp(-((E / theta) ** 2))
        coef = np.linalg.solve(R + lam * np.eye(n_obs), y)
        M = np.exp(-(((test[:, None] - D[None, :]) / theta) ** 2))
        return M @ coef

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (title, D, yy) in zip(
        axes,
        [
            ("kernel ridge regression (equi-spaced)", D_equi, y),
            ("kernel ridge regression (D-optimal)", D_opt, y_opt),
        ],
    ):
        ax.plot(test, true, color="black", linewidth=3, linestyle="--")
        ax.scatter(D, yy, s=40, color="blue")
        pred = _kernel_ridge_cv(D, yy, test)
        ax.plot(test, pred, color="green", linewidth=3)
        ax.set_ylim(true.min() - 0.25, true.max() + 0.25)
        ax.set_title(title, fontsize=12)
        ax.legend(["truth", "prediction"], loc="lower right", frameon=False)
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "kernel_ridge_regression.png"))

    # ------------------------------------------------------------------
    # Gaussian process regression (equi-spaced vs D-optimal) via rkriging
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch2_gp_regression.R",
            inputs={
                "data.csv": pd.DataFrame({"placeholder": [0]}),
                "params.csv": pd.DataFrame({"seed": [5], "n": [10], "re": [2]}),
            },
            args=[],
            required_packages=["rkriging", "AlgDesign"],
        )

        equi_data = out["equi_data.csv"]
        equi_pred = out["equi_pred.csv"]
        dopt_data = out["dopt_data.csv"]
        dopt_pred = out["dopt_pred.csv"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Equi-spaced GP regression
        ax = axes[0]
        ax.plot(test, true, color="black", linewidth=3, linestyle="--")
        ax.scatter(equi_data["x"].values, equi_data["y"].values, s=40, color="blue")
        mean = equi_pred["mean"].values
        sd = equi_pred["sd"].values
        ax.fill_between(test, mean - 2 * sd, mean + 2 * sd, color="pink", alpha=1.0, linewidth=0)
        ax.plot(test, mean, color="green", linewidth=3)
        ax.set_ylim(true.min() - 0.25, true.max() + 0.25)
        ax.set_title("Gaussian process regression (equi-spaced)", fontsize=12)
        ax.legend(["truth", "prediction"], loc="lower right", frameon=False)

        # D-optimal GP regression
        ax = axes[1]
        ax.plot(test, true, color="black", linewidth=3, linestyle="--")
        ax.scatter(dopt_data["x"].values, dopt_data["y"].values, s=40, color="blue")
        mean = dopt_pred["mean"].values
        sd = dopt_pred["sd"].values
        ax.fill_between(test, mean - 2 * sd, mean + 2 * sd, color="pink", alpha=1.0, linewidth=0)
        ax.plot(test, mean, color="green", linewidth=3)
        ax.set_ylim(true.min() - 0.25, true.max() + 0.25)
        ax.set_title("Gaussian process regression (D-optimal)", fontsize=12)
        ax.legend(["truth", "prediction"], loc="lower right", frameon=False)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "gp_regression.png"))

    return out_dir, paths


def main(argv: Sequence[str] | None = None) -> int:
    generate_figures(use_r=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
