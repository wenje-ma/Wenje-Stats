"""
Python port of `R files/Ch9.R` (Model Calibration).

Sections covered:
- 9.1.1 Analytical Models (SO2 nonlinear regression, FIM)
- 9.1.2 Computationally Expensive Models (Kriging functional output)
- 9.2.1 Parameter Uncertainties (Locally D-optimal vs Robust design)
- 9.2.2 Model Uncertainties (Model prediction, MaxPro augmentation)

Python-first policy:
- Plotting is done in Python (matplotlib).
- R is invoked for: ICAOD (optimal design), rkriging (functional Kriging),
  support (support points), MaxPro (augmentation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d

from pyexpdes.core.paths import figures_dir, datasets_dir
from pyexpdes.core.style import apply_base_style
from pyexpdes.core.r_utils import run_wrapper


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Section 9.1.1: SO2 Analytical Model
# ---------------------------------------------------------------------------

def _h_so2(u: np.ndarray, eta: np.ndarray) -> float:
    """
    SO2 model function.

    Args:
        u: [x1, x2, dt, y0] input vector
        eta: [beta0, beta1, m] parameters

    Returns:
        Predicted y1 value
    """
    x1, x2, dt, y0 = u[0], u[1], u[2], u[3]
    beta0, beta1, m = eta[0], eta[1], eta[2]

    rate = beta0 + beta1 * x1 + x2
    frac = (beta0 + beta1 * x1) / rate
    val = y0 + (frac - y0) * (1 - np.exp(-rate * dt / m))
    return val


def _gradient_so2(x: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """
    Gradient of SO2 model w.r.t. parameters.

    Args:
        x: [x1, x2, dt, y0] input vector
        eta: [beta0, beta1, m] parameters

    Returns:
        Gradient [d/d_beta0, d/d_beta1, d/d_m]
    """
    x1, x2, dt, y0 = x[0], x[1], x[2], x[3]
    beta0, beta1, m = eta[0], eta[1], eta[2]

    rate = beta0 + beta1 * x1 + x2
    frac = (beta0 + beta1 * x1) / rate
    term1 = frac - y0
    term2 = np.exp(-rate * dt / m)

    v1 = term1 * term2 * dt / m + x2 / rate**2 * (1 - term2)
    v2 = term1 * term2 * x1 * dt / m + x1 * x2 / rate**2 * (1 - term2)
    v3 = -term1 * term2 * rate * dt / m**2

    return np.array([v1, v2, v3])


def _fit_so2_model(so2_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit SO2 model using least squares.

    Returns:
        (fitted_params, sensitivity_matrix)
    """
    y = so2_data["y1"].values
    X = so2_data[["x1", "x2", "dt", "y0"]].values

    def sse(eta):
        yhat = np.array([_h_so2(X[i], eta) for i in range(len(X))])
        return np.sum((y - yhat) ** 2)

    result = minimize(sse, x0=[0, 1, 1000], method="Nelder-Mead")
    eta_hat = result.x

    # Compute sensitivity matrix
    S = np.array([_gradient_so2(X[i], eta_hat) for i in range(len(X))])

    return eta_hat, S


# ---------------------------------------------------------------------------
# Figure Generation
# ---------------------------------------------------------------------------

def generate_figures(*, use_r: bool = True) -> tuple[Path, list[Path]]:
    """
    Generate Chapter 9 figures.

    Returns (output_dir, [figure_paths...]).
    """
    apply_base_style()
    out_dir = figures_dir("ch09")
    paths: list[Path] = []

    # ------------------------------------------------------------------
    # Section 9.1.1: SO2 Model - Fit and FIM Analysis (no figure)
    # ------------------------------------------------------------------
    so2_path = datasets_dir() / "so2.csv"
    if so2_path.exists():
        so2_data = pd.read_csv(so2_path)
        eta_hat, S = _fit_so2_model(so2_data)
        # FIM = S.T @ S
        # This section matches the R code analysis of SO2 model fitting

    # ------------------------------------------------------------------
    # Section 9.1.1: ICAOD Locally and Bayesian D-optimal Design
    # Two-factor exponential model: eta0 + exp(-eta1*x1) + exp(-eta2*x2)
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch9_icaod_locally_bayes.R",
            inputs={
                "params.csv": pd.DataFrame({
                    "seed": [1],
                    "m": [4],  # Number of design points
                })
            },
            args=[],
            required_packages=["ICAOD"],
        )

        D_local = out["local_design.csv"][["x1", "x2"]].values
        w_local = out["local_design.csv"]["weight"].values
        D_bayes = out["bayes_design.csv"][["x1", "x2"]].values
        w_bayes = out["bayes_design.csv"]["weight"].values
        summary = out["design_summary.csv"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Locally D-optimal design
        ax = axes[0]
        for i, (x1, x2, w) in enumerate(zip(D_local[:, 0], D_local[:, 1], w_local)):
            size = 200 * w / w_local.max() + 100
            ax.scatter(x1, x2, s=size, color="blue", zorder=5, edgecolors="black")
            ax.annotate(f"{w:.2f}", (x1 + 0.03, x2 + 0.03), fontsize=10, color="red")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(r"$x_1$", fontsize=14)
        ax.set_ylabel(r"$x_2$", fontsize=14)
        ax.set_title("Locally D-optimal Design", fontsize=16)
        ax.grid(True, alpha=0.3)

        # Right: Bayesian D-optimal design
        ax = axes[1]
        for i, (x1, x2, w) in enumerate(zip(D_bayes[:, 0], D_bayes[:, 1], w_bayes)):
            size = 200 * w / w_bayes.max() + 100
            ax.scatter(x1, x2, s=size, color="green", zorder=5, edgecolors="black")
            ax.annotate(f"{w:.2f}", (x1 + 0.03, x2 + 0.03), fontsize=10, color="red")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(r"$x_1$", fontsize=14)
        ax.set_ylabel(r"$x_2$", fontsize=14)
        ax.set_title("Pseudo-Bayesian D-optimal Design", fontsize=16)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "fig9_0_icaod_designs.png"))

    # ------------------------------------------------------------------
    # Figure 9.1: Functional Output and Optimal Design (Kriging)
    # ------------------------------------------------------------------
    if use_r:
        functional_dir = datasets_dir() / "functional"

        out = run_wrapper(
            "ch9_kriging_functional.R",
            inputs={
                "params.csv": pd.DataFrame({
                    "seed": [1],
                    "functional_dir": [str(functional_dir)],
                })
            },
            args=[],
            required_packages=["rkriging", "FNN", "ICAOD", "numDeriv"],
        )

        # Load functional data for plotting
        results = pd.read_csv(functional_dir / "results.csv")
        pdet = np.concatenate([
            np.linspace(0, 5000, 2001),
            np.linspace(5000, 120000, 4601)[1:]
        ])

        # Read simulation outputs
        N = 50
        Y = np.zeros((N, len(pdet)))
        for i in range(N):
            Y[i] = np.loadtxt(functional_dir / f"run{i+1}.txt")

        # Get optimal design points and predictions
        D_opt = out["optimal_design.csv"]["time"].values
        pred = out["prediction.csv"]["pred"].values
        tsamp = out["prediction.csv"]["time"].values
        y_D = out["optimal_design.csv"]["y_pred"].values

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Functional output
        ax = axes[0]
        for i in range(N):
            ax.plot(pdet, Y[i], color="gray", alpha=0.5, linewidth=0.5)
        ax.set_xlabel("time (seconds)", fontsize=14)
        ax.set_ylabel("mass uptake", fontsize=14)
        ax.set_title("functional output", fontsize=16)

        # Right: Optimal design
        ax = axes[1]
        ax.plot(tsamp, pred, color="black", linewidth=2)
        ax.scatter(D_opt * 120000, y_D, s=200, color="blue",
                   edgecolors="black", linewidths=2, zorder=5)
        ax.set_xlabel("time (seconds)", fontsize=14)
        ax.set_ylabel("mass uptake", fontsize=14)
        ax.set_title("optimal design", fontsize=16)
        ax.set_ylim(Y.min() - 0.01, Y.max() + 0.01)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "fig9_1_functional_optimal.png"))

    # ------------------------------------------------------------------
    # Figure 9.2: Locally D-optimal vs Robust Design
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch9_robust_design.R",
            inputs={
                "params.csv": pd.DataFrame({
                    "seed": [1],
                    "N": [20],  # Number of parameter samples
                    "n_select": [8],  # Number of design points to select
                })
            },
            args=[],
            required_packages=["support"],
        )

        L = out["local_design.csv"].values
        D_cand = out["candidate_design.csv"].values
        D_robust = out["robust_design.csv"].values

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: Locally D-optimal design
        ax = axes[0]
        ax.scatter(L[:, 0], L[:, 1], s=300, color="blue", zorder=5)
        ax.text(L[0, 0] + 0.03, L[0, 1] + 0.03, "2", fontsize=16, color="red")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$x_1$", fontsize=14)
        ax.set_ylabel(r"$x_2$", fontsize=14)
        ax.set_title("Locally D-optimal Design", fontsize=16)

        # Right: Robust design
        ax = axes[1]
        ax.scatter(D_cand[:, 0], D_cand[:, 1], s=30, color="gray", alpha=0.5)
        ax.scatter(D_robust[:, 0], D_robust[:, 1], s=300, color="blue", zorder=5)
        ax.text(0.03, 0.03, "2", fontsize=16, color="red")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$x_1$", fontsize=14)
        ax.set_ylabel(r"$x_2$", fontsize=14)
        ax.set_title("Robust Design", fontsize=16)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "fig9_2_robust_design.png"))

    # ------------------------------------------------------------------
    # Figure 9.3: Model Prediction vs Real Data
    # ------------------------------------------------------------------
    functional_dir = datasets_dir() / "functional"

    # Load best prediction (run51)
    best = np.loadtxt(functional_dir / "run51.txt")
    pdet = np.concatenate([
        np.linspace(0, 5000, 2001),
        np.linspace(5000, 120000, 4601)[1:]
    ])

    # Load experimental data
    sorp = pd.read_csv(functional_dir / "exp_sorption_130c_8.7torr_483nm.csv")
    desorp = pd.read_csv(functional_dir / "exp_desorption_130c_8.7torr_483nm.csv")

    # Correct desorption offset (from original R code)
    correct = 35358.89275 - 34644.0896 - 1
    y_exp = np.concatenate([sorp.iloc[:, 1].values, desorp.iloc[:, 1].values + correct])
    t_exp = np.concatenate([sorp.iloc[:, 0].values**2, desorp.iloc[:, 0].values**2 + 62973])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.sqrt(t_exp), y_exp, color="black", linewidth=2, linestyle="--",
            label="measured")
    ax.plot(np.sqrt(pdet), best, color="red", linewidth=2, label="predicted")
    ax.set_xlabel(r"$\sqrt{t}$", fontsize=14)
    ax.set_ylabel("mass uptake", fontsize=14)
    ax.set_title("model prediction vs real data", fontsize=16)
    ax.legend(loc="lower right", fontsize=12, frameon=False)

    fig.tight_layout()
    paths.append(_save(fig, out_dir / "fig9_3_model_prediction.png"))

    # ------------------------------------------------------------------
    # Figure 9.4: MaxPro Augmentation Design
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch9_augment_design.R",
            inputs={
                "params.csv": pd.DataFrame({
                    "seed": [1],
                    "N": [20],  # Number of parameter samples
                    "n_initial": [4],  # Initial design points
                    "n_new": [4],  # New points to add
                    "n_cand": [100],  # Candidate points
                })
            },
            args=[],
            required_packages=["support", "MaxPro"],
        )

        D_cand = out["candidate_design.csv"].values
        D_initial = out["initial_design.csv"].values
        D_new = out["new_design.csv"].values

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(D_cand[:, 0], D_cand[:, 1], s=30, color="gray", alpha=0.5)
        ax.scatter(D_initial[:, 0], D_initial[:, 1], s=300, color="blue", zorder=5,
                   label="initial")
        ax.scatter(D_new[:, 0], D_new[:, 1], s=200, color="red", marker="x",
                   linewidths=3, zorder=6, label="new")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$x_1$", fontsize=14)
        ax.set_ylabel(r"$x_2$", fontsize=14)
        ax.set_title("Robust Design with Augmentation", fontsize=16)
        ax.legend(loc="upper right", fontsize=12, frameon=False)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "fig9_4_augment_design.png"))

    return out_dir, paths


def main(argv: Sequence[str] | None = None) -> int:
    generate_figures(use_r=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
