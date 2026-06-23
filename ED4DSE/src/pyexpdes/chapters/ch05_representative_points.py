"""
Python port of `R files/Ch5.R` (Representative Points).

Sections covered:
- 5.1 Uniform Design (uniformity, Sobol, discrepancy)
- 5.2 Nonuniform distributions (transformation, support points, uncertainty propagation)
- PQMC (Section 5.2.4) is excluded due to external dependency.

Python-first policy:
- Plotting is done in Python (matplotlib).
- R is invoked for specialized packages: SFDesign, spacefillr, support, adaptMCMC.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma as sp_gamma

from pyexpdes.core.paths import figures_dir
from pyexpdes.core.style import apply_base_style
from pyexpdes.core.r_utils import run_wrapper


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _borehole(x: np.ndarray) -> float:
    """
    Borehole function (8 variables).
    Input x should be in [0,1]^8, will be scaled internally.
    """
    lower = np.array([0.05, 100, 63070, 990, 63.1, 700, 1120, 9855])
    upper = np.array([0.15, 50000, 115600, 1110, 116, 820, 1680, 12045])
    x_scaled = lower + x * (upper - lower)

    rw, r, Tu, Hu, Tl, Hl, L, Kw = x_scaled
    log_ratio = np.log(r / rw)
    numer = 2 * np.pi * Tu * (Hu - Hl)
    denom = log_ratio * (1 + 2 * L * Tu / (log_ratio * rw**2 * Kw) + Tu / Tl)
    return float(numer / denom)


def generate_figures(*, use_r: bool = True) -> tuple[Path, list[Path]]:
    """
    Generate Chapter 5 figures.

    Returns (output_dir, [figure_paths...]).
    """
    apply_base_style()
    out_dir = figures_dir("ch05")
    paths: list[Path] = []

    # ------------------------------------------------------------------
    # Figure 5.1: Uniformity - random vs optimal design (1D)
    # ------------------------------------------------------------------
    n = 7
    rng = np.random.default_rng(1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Random design
    x_rand = np.sort(rng.random(n))
    ax = axes[0]
    ax.scatter(x_rand, np.arange(1, n + 1) / n, s=150, color="blue")
    # Step function (empirical CDF)
    ax.step([0] + list(x_rand) + [1], [0] + list(np.arange(1, n + 1) / n) + [1],
            where="post", color="red", linewidth=2)
    ax.plot([0, 1], [0, 1], color="green", linewidth=2, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("distribution function", fontsize=14)
    ax.set_title("random", fontsize=16)

    # Optimal design
    x_opt = (np.arange(1, n + 1) - 0.5) / n
    ax = axes[1]
    ax.scatter(x_opt, np.arange(1, n + 1) / n, s=150, color="blue")
    ax.step([0] + list(x_opt) + [1], [0] + list(np.arange(1, n + 1) / n) + [1],
            where="post", color="red", linewidth=2)
    ax.plot([0, 1], [0, 1], color="green", linewidth=2, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("distribution function", fontsize=14)
    ax.set_title("optimal", fontsize=16)

    fig.tight_layout()
    paths.append(_save(fig, out_dir / "uniformity_1d.png"))

    # ------------------------------------------------------------------
    # Figure 5.2: Monte Carlo vs Sobol vs Uniform design (2D)
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch5_uniform_design.R",
            inputs={"params.csv": pd.DataFrame({"n": [20], "p": [2], "seed": [6]})},
            args=[],
            required_packages=["SFDesign", "spacefillr"],
        )
        R = out["random_design.csv"].to_numpy()
        S = out["sobol_design.csv"].to_numpy()
        D = out["uniform_design.csv"].to_numpy()
        crits = out["uniform_crit.csv"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        designs = [
            ("Monte Carlo", R, crits[crits["method"] == "random"]["crit"].values[0]),
            ("Sobol Sequence", S, crits[crits["method"] == "sobol"]["crit"].values[0]),
            ("Uniform Design", D, crits[crits["method"] == "uniform"]["crit"].values[0]),
        ]
        for ax, (title, pts, crit) in zip(axes, designs):
            ax.scatter(pts[:, 0], pts[:, 1], s=150, color="blue")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel(r"$x_1$", fontsize=14)
            ax.set_ylabel(r"$x_2$", fontsize=14)
            ax.set_title(f"{title}\n(crit={crit:.4f})", fontsize=14)
            ax.set_aspect("equal")

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "uniform_design_2d.png"))

    # ------------------------------------------------------------------
    # Figure 5.3: Discrepancy and Integration simulation
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch5_uniform_simulation.R",
            inputs={"params.csv": pd.DataFrame({
                "n_seq": ["10,20,30,40,50,60,70,80,90,100"],
                "n_rep": [30],
                "p": [2],
                "seed": [1]
            })},
            args=[],
            required_packages=["SFDesign", "spacefillr"],
        )
        results = out["simulation_results.csv"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Discrepancy plot
        ax = axes[0]
        n_vals = sorted(results["n"].unique())
        for method, col, marker in [("rand_crit", "C0", "o"), ("sob_crit", "C1", "s"), ("uni_crit", "C2", "^")]:
            medians = [results[results["n"] == n][method].median() for n in n_vals]
            lowers = [results[results["n"] == n][method].quantile(0.05) for n in n_vals]
            uppers = [results[results["n"] == n][method].quantile(0.95) for n in n_vals]
            ax.plot(n_vals, medians, marker=marker, color=col, linewidth=2)
            ax.fill_between(n_vals, lowers, uppers, alpha=0.3, color=col)

        ax.set_xlabel("n", fontsize=14)
        ax.set_ylabel("discrepancy", fontsize=14)
        ax.set_title("Discrepancy", fontsize=16)
        ax.legend(["MC", "Sobol", "Uniform"], loc="upper right", frameon=False)

        # Integration plot
        ax = axes[1]
        for method, col, marker in [("rand_int", "C0", "o"), ("sob_int", "C1", "s"), ("uni_int", "C2", "^")]:
            medians = [results[results["n"] == n][method].median() for n in n_vals]
            lowers = [results[results["n"] == n][method].quantile(0.05) for n in n_vals]
            uppers = [results[results["n"] == n][method].quantile(0.95) for n in n_vals]
            ax.plot(n_vals, medians, marker=marker, color=col, linewidth=2)
            ax.fill_between(n_vals, lowers, uppers, alpha=0.3, color=col)

        ax.set_xlabel("n", fontsize=14)
        ax.set_ylabel("integral", fontsize=14)
        ax.set_title("Integration", fontsize=16)
        ax.legend(["MC", "Sobol", "Uniform"], loc="upper right", frameon=False)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "discrepancy_integration.png"))

    # ------------------------------------------------------------------
    # Figure 5.4: Transformation methods (Normal and Gamma)
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch5_uniform_design.R",
            inputs={"params.csv": pd.DataFrame({"n": [50], "p": [2], "seed": [6]})},
            args=[],
            required_packages=["SFDesign", "spacefillr"],
        )
        d = out["uniform_design.csv"].to_numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Normal distribution
        D_norm = norm.ppf(d)
        ax = axes[0]
        N_plot = 100
        p1 = np.linspace(-3, 3, N_plot)
        p2 = np.linspace(-3, 3, N_plot)
        P1, P2 = np.meshgrid(p1, p2)
        fc = np.exp(-0.5 * (P1**2 + P2**2)) / (2 * np.pi)
        ax.imshow(fc, origin="lower", extent=(-3, 3, -3, 3), cmap="cool", aspect="auto")
        ax.contour(P1, P2, fc, levels=10, colors="black", linewidths=0.5)
        ax.scatter(D_norm[:, 0], D_norm[:, 1], s=80, color="blue")
        ax.set_xlabel(r"$x_1$", fontsize=14)
        ax.set_ylabel(r"$x_2$", fontsize=14)
        ax.set_title("Normal distribution", fontsize=16)

        # Gamma distribution
        D_gamma = sp_gamma.ppf(d, 2, scale=1)
        ax = axes[1]
        p1 = np.linspace(0, 6, N_plot)
        p2 = np.linspace(0, 6, N_plot)
        P1, P2 = np.meshgrid(p1, p2)
        fc = sp_gamma.pdf(P1, 2, scale=1) * sp_gamma.pdf(P2, 2, scale=1)
        ax.imshow(fc, origin="lower", extent=(0, 6, 0, 6), cmap="cool", aspect="auto")
        ax.contour(P1, P2, fc, levels=10, colors="black", linewidths=0.5)
        ax.scatter(D_gamma[:, 0], D_gamma[:, 1], s=80, color="blue")
        ax.set_xlabel(r"$x_1$", fontsize=14)
        ax.set_ylabel(r"$x_2$", fontsize=14)
        ax.set_title("Gamma distribution", fontsize=16)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "transformation_methods.png"))

    # ------------------------------------------------------------------
    # Figure 5.5: Rosenblatt transformation (correlated normal)
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch5_uniform_design.R",
            inputs={"params.csv": pd.DataFrame({"n": [50], "p": [2], "seed": [6]})},
            args=[],
            required_packages=["SFDesign", "spacefillr"],
        )
        d = out["uniform_design.csv"].to_numpy()

        # Rosenblatt transformation for correlated normal (rho=0.5)
        rho = 0.5
        x1 = norm.ppf(d[:, 0])
        x2 = norm.ppf(d[:, 1], loc=rho * x1, scale=np.sqrt(1 - rho**2))
        D = np.column_stack([x1, x2])

        fig, ax = plt.subplots(figsize=(6, 6))
        N_plot = 100
        p1 = np.linspace(-3, 3, N_plot)
        p2 = np.linspace(-3, 3, N_plot)
        P1, P2 = np.meshgrid(p1, p2)

        # Bivariate normal density with rho=0.5
        det = 1 - rho**2
        fc = np.exp(-0.5 * (P1**2 - 2 * rho * P1 * P2 + P2**2) / det) / (2 * np.pi * np.sqrt(det))

        ax.imshow(fc, origin="lower", extent=(-3, 3, -3, 3), cmap="cool", aspect="auto")
        ax.contour(P1, P2, fc, levels=10, colors="black", linewidths=0.5)
        ax.scatter(D[:, 0], D[:, 1], s=80, color="blue")
        ax.set_xlabel(r"$x_1$", fontsize=14)
        ax.set_ylabel(r"$x_2$", fontsize=14)
        ax.set_title("Correlated normal distribution", fontsize=14)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "rosenblatt_transformation.png"))

    # ------------------------------------------------------------------
    # Figure 5.6: Support points (Normal and Gamma)
    # ------------------------------------------------------------------
    if use_r:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Normal support points
        out = run_wrapper(
            "ch5_support_points.R",
            inputs={"params.csv": pd.DataFrame({"n": [50], "p": [2], "dist_type": ["normal"], "seed": [1]})},
            args=[],
            required_packages=["support", "mnormt"],
        )
        D = out["support_points.csv"].to_numpy()
        ax = axes[0]
        N_plot = 100
        p1 = np.linspace(-3, 3, N_plot)
        p2 = np.linspace(-3, 3, N_plot)
        P1, P2 = np.meshgrid(p1, p2)
        fc = np.exp(-0.5 * (P1**2 + P2**2)) / (2 * np.pi)
        ax.imshow(fc, origin="lower", extent=(-3, 3, -3, 3), cmap="cool", aspect="auto")
        ax.contour(P1, P2, fc, levels=10, colors="black", linewidths=0.5)
        ax.scatter(D[:, 0], D[:, 1], s=80, color="blue")
        ax.set_xlabel(r"$x_1$", fontsize=14)
        ax.set_ylabel(r"$x_2$", fontsize=14)
        ax.set_title("Normal distribution", fontsize=16)

        # Gamma support points
        out = run_wrapper(
            "ch5_support_points.R",
            inputs={"params.csv": pd.DataFrame({"n": [50], "p": [2], "dist_type": ["gamma"], "seed": [1]})},
            args=[],
            required_packages=["support", "mnormt"],
        )
        D = out["support_points.csv"].to_numpy()
        ax = axes[1]
        p1 = np.linspace(0, 6, N_plot)
        p2 = np.linspace(0, 6, N_plot)
        P1, P2 = np.meshgrid(p1, p2)
        fc = sp_gamma.pdf(P1, 2, scale=1) * sp_gamma.pdf(P2, 2, scale=1)
        ax.imshow(fc, origin="lower", extent=(0, 6, 0, 6), cmap="cool", aspect="auto")
        ax.contour(P1, P2, fc, levels=10, colors="black", linewidths=0.5)
        ax.scatter(D[:, 0], D[:, 1], s=80, color="blue")
        ax.set_xlabel(r"$x_1$", fontsize=14)
        ax.set_ylabel(r"$x_2$", fontsize=14)
        ax.set_title("Gamma distribution", fontsize=16)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "support_points.png"))

    # ------------------------------------------------------------------
    # Figure 5.7: Support points (correlated normal + banana)
    # ------------------------------------------------------------------
    if use_r:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Correlated normal
        out = run_wrapper(
            "ch5_support_points.R",
            inputs={"params.csv": pd.DataFrame({"n": [50], "p": [2], "dist_type": ["correlated_normal"], "seed": [1]})},
            args=[],
            required_packages=["support", "mnormt"],
        )
        D = out["support_points.csv"].to_numpy()
        ax = axes[0]
        rho = 0.5
        N_plot = 100
        p1 = np.linspace(-3, 3, N_plot)
        p2 = np.linspace(-3, 3, N_plot)
        P1, P2 = np.meshgrid(p1, p2)
        det = 1 - rho**2
        fc = np.exp(-0.5 * (P1**2 - 2 * rho * P1 * P2 + P2**2) / det) / (2 * np.pi * np.sqrt(det))
        ax.imshow(fc, origin="lower", extent=(-3, 3, -3, 3), cmap="cool", aspect="auto")
        ax.contour(P1, P2, fc, levels=10, colors="black", linewidths=0.5)
        ax.scatter(D[:, 0], D[:, 1], s=80, color="blue")
        ax.set_xlabel(r"$x_1$", fontsize=14)
        ax.set_ylabel(r"$x_2$", fontsize=14)
        ax.set_title("Correlated normal distribution", fontsize=14)

        # Banana-shaped density
        out = run_wrapper(
            "ch5_banana_mcmc.R",
            inputs={"params.csv": pd.DataFrame({"n": [50], "p": [2], "seed": [1], "n_mcmc": [100000]})},
            args=[],
            required_packages=["support", "adaptMCMC"],
        )
        D = out["support_points.csv"].to_numpy()
        grid = out["banana_grid.csv"]
        ax = axes[1]

        # Reconstruct grid: R expand.grid(p1, p2) varies p1 first -> reshape (len_p2, len_p1)
        p1 = np.sort(grid["p1"].unique())
        p2 = np.sort(grid["p2"].unique())
        fc = grid["fc"].to_numpy().reshape(len(p2), len(p1), order="C")
        ax.imshow(fc, origin="lower", extent=(0, 1, 0, 1), cmap="Blues", aspect="auto")
        ax.contour(p1, p2, fc, levels=10, colors="black", linewidths=0.5)
        ax.scatter(D[:, 0], D[:, 1], s=60, color="blue")
        ax.set_xlabel(r"$\theta_1$", fontsize=14)
        ax.set_ylabel(r"$\theta_2$", fontsize=14)
        ax.set_title("Banana-shaped density", fontsize=14)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "support_points_advanced.png"))

    # ------------------------------------------------------------------
    # Figure 5.8: Uncertainty propagation (borehole function)
    # ------------------------------------------------------------------
    if use_r:
        # Generate population samples for true distribution
        rng = np.random.default_rng(1)
        N = 100000
        p = 8
        lower = np.array([0.05, 100, 63070, 990, 63.1, 700, 1120, 9855])
        upper = np.array([0.15, 50000, 115600, 1110, 116, 820, 1680, 12045])

        # X1 ~ N(0.1, 0.01618^2), X2 ~ LogNormal
        X1 = 0.1 + 0.01618 * rng.normal(size=N)
        X2 = np.exp(7.71 + 1.0056 * rng.normal(size=N))
        X = np.zeros((N, p))
        X[:, 0] = X1
        X[:, 1] = X2
        X[:, 2:] = rng.random((N, p - 2))
        X[:, 2:] = X[:, 2:] * (upper[2:] - lower[2:]) + lower[2:]

        # Scale X1, X2 to [0,1] for borehole function
        X_scaled = np.zeros((N, p))
        X_scaled[:, 0] = (X[:, 0] - lower[0]) / (upper[0] - lower[0])
        X_scaled[:, 1] = (X[:, 1] - lower[1]) / (upper[1] - lower[1])
        X_scaled[:, 2:] = (X[:, 2:] - lower[2:]) / (upper[2:] - lower[2:])
        X_scaled = np.clip(X_scaled, 0, 1)
        true_y = np.array([_borehole(x) for x in X_scaled])

        # Monte Carlo sample
        n = 100
        idx_mc = rng.choice(N, n, replace=False)
        mc_y = true_y[idx_mc]

        # Support points via R wrapper
        # Standardize samples for support points
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        Z = (X - mu) / sigma

        out = run_wrapper(
            "ch5_support_points.R",
            inputs={
                "params.csv": pd.DataFrame({"n": [n], "p": [p], "dist_type": ["from_samples"], "seed": [1]}),
                "samples.csv": pd.DataFrame(Z),
            },
            args=[],
            required_packages=["support", "mnormt"],
        )
        D_sp = out["support_points.csv"].to_numpy()
        D_sp = D_sp * sigma + mu  # Unstandardize

        # Scale to [0,1] for borehole
        D_sp_scaled = np.zeros_like(D_sp)
        D_sp_scaled[:, 0] = (D_sp[:, 0] - lower[0]) / (upper[0] - lower[0])
        D_sp_scaled[:, 1] = (D_sp[:, 1] - lower[1]) / (upper[1] - lower[1])
        D_sp_scaled[:, 2:] = (D_sp[:, 2:] - lower[2:]) / (upper[2:] - lower[2:])
        D_sp_scaled = np.clip(D_sp_scaled, 0, 1)
        sp_y = np.array([_borehole(x) for x in D_sp_scaled])

        fig, ax = plt.subplots(figsize=(8, 6))
        from scipy.stats import gaussian_kde

        # True distribution
        kde_true = gaussian_kde(true_y)
        xs = np.linspace(0, 200, 500)
        ax.plot(xs, kde_true(xs), color="black", linewidth=3, linestyle=":", label="truth")

        # MC
        kde_mc = gaussian_kde(mc_y)
        ax.plot(xs, kde_mc(xs), color="red", linewidth=3, linestyle="--", label="MC")

        # Support points
        kde_sp = gaussian_kde(sp_y)
        ax.plot(xs, kde_sp(xs), color="green", linewidth=3, label="support points")

        ax.set_xlabel("y", fontsize=14)
        ax.set_ylabel("density", fontsize=14)
        ax.set_title("output distribution", fontsize=16)
        ax.legend(loc="upper right", frameon=False, fontsize=12)
        ax.set_xlim(0, 200)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "uncertainty_propagation.png"))

    return out_dir, paths


def main(argv: Sequence[str] | None = None) -> int:
    generate_figures(use_r=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

