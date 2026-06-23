"""
Python port of `R files/Ch6.R` (Screening Designs).

Sections covered:
- 6.1 Sensitivity Analysis (variance-based, derivative-based)
- 6.2 Morris screening design
- 6.3 MOFAT designs

Python-first policy:
- Plotting is done in Python (matplotlib).
- R is invoked for: sensitivity (soboljansen, delsa, morris), MOFAT.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    Generate Chapter 6 figures.

    Returns (output_dir, [figure_paths...]).
    """
    apply_base_style()
    out_dir = figures_dir("ch06")
    paths: list[Path] = []

    p = 8  # borehole has 8 variables

    # ------------------------------------------------------------------
    # Figure 6.1: Sobol' indices + Main effects
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch6_sobol_sensitivity.R",
            inputs={"params.csv": pd.DataFrame({"p": [8], "m": [10000], "seed": [1]})},
            args=[],
            required_packages=["sensitivity", "SFDesign"],
        )
        sobol_idx = out["sobol_indices.csv"]
        main_effects = out["main_effects.csv"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Stacked bar plot for Sobol indices
        ax = axes[0]
        variables = sobol_idx["variable"].values
        first_order = sobol_idx["first_order"].values
        total = sobol_idx["total"].values
        interaction = total - first_order

        x_pos = np.arange(len(variables))
        ax.bar(x_pos, first_order, label="first order", color="lightblue")
        ax.bar(x_pos, interaction, bottom=first_order, label="total-first order", color="pink")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(variables, fontsize=12)
        ax.set_ylabel("total Sobol' index", fontsize=14)
        ax.legend(loc="upper right", frameon=False, fontsize=12)

        # Main effects plot
        ax = axes[1]
        x_ev = main_effects["x"].values
        for j in range(1, 9):
            ax.plot(x_ev, main_effects[f"x{j}"].values, linewidth=2, label=f"x{j}")
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("main effects", fontsize=14)
        ax.legend(loc="upper left", frameon=False, fontsize=10, ncol=2)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "sobol_main_effects.png"))

    # ------------------------------------------------------------------
    # Figure 6.2: DELSA boxplot (derivative-based sensitivity)
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch6_delsa.R",
            inputs={"params.csv": pd.DataFrame({"p": [8], "m": [4], "n_rep": [100], "seed": [1]})},
            args=[],
            required_packages=["sensitivity", "SFDesign"],
        )
        delsa_results = out["delsa_results.csv"]

        fig, ax = plt.subplots(figsize=(8, 6))
        data = [delsa_results[f"x{j}"].values for j in range(1, 9)]
        bp = ax.boxplot(data, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
        ax.set_xticklabels([f"x{j}" for j in range(1, 9)], fontsize=12)
        ax.set_ylabel(r"$\bar{\nu}$", fontsize=14)
        ax.set_title("Derivative-based sensitivity measure", fontsize=14)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "delsa_boxplot.png"))

    # ------------------------------------------------------------------
    # Figure 6.3: Variance-based vs Derivative-based comparison
    # This uses a different test function (sum of sines)
    # ------------------------------------------------------------------
    p_test = 4

    if use_r:
        # Call R wrapper for actual DELSA and Sobol' calculations
        out = run_wrapper(
            "ch6_variance_vs_derivative.R",
            inputs={"params.csv": pd.DataFrame({"p": [p_test], "seed": [1]})},
            args=[],
            required_packages=["sensitivity", "SFDesign"],
        )
        comparison = out["comparison.csv"]
        nu = comparison["nu"].values
        T = comparison["T"].values

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: function curves
        ax = axes[0]
        x_vals = np.linspace(0, 1, 200)
        for k in range(1, p_test + 1):
            y_vals = np.sin((k**2 + 1) * np.pi * x_vals) / k
            ax.plot(x_vals, y_vals, linewidth=2, label=f"x{k}")
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.legend(loc="upper right", frameon=False, fontsize=12)

        # Right: comparison plot with actual computed values
        ax = axes[1]
        x_pos = np.arange(1, p_test + 1)
        ax.scatter(x_pos, T, s=150, color="blue", marker="o", label="T (variance)")
        ax.scatter(x_pos, nu, s=150, color="red", marker="x", linewidths=3, label=r"$\bar{\nu}$ (derivative)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"x{j}" for j in range(1, p_test + 1)], fontsize=12)
        ax.set_ylabel("sensitivity measure", fontsize=14)
        ax.set_title("Variance-based vs Derivative-based", fontsize=14)
        ax.legend(loc="upper right", frameon=False, fontsize=12)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "variance_vs_derivative.png"))

    # ------------------------------------------------------------------
    # Figure 6.4: Morris design visualization (pure Python)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(10)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Strict OFAT
    x_grid = (np.arange(4)) / 3
    A = np.column_stack([x_grid, rng.permutation(x_grid)])
    delta = 1 / 3

    C1 = A.copy()
    C2 = A.copy()
    for i in range(4):
        # Change x1
        C1[i, 0] = A[i, 0] + delta
        if C1[i, 0] > 1:
            C1[i, 0] = A[i, 0] - delta
        # Then change x2
        C2[i, 0] = C1[i, 0]
        C2[i, 1] = C1[i, 1] + delta
        if C2[i, 1] > 1:
            C2[i, 1] = C1[i, 1] - delta

    ax = axes[0]
    ax.scatter(A[:, 0], A[:, 1], s=150, color="black", marker="o", linewidths=2)
    ax.scatter(C1[:, 0], C1[:, 1], s=150, color="red", marker="^", linewidths=2)
    ax.scatter(C2[:, 0], C2[:, 1], s=150, color="green", marker="s", linewidths=2)
    for i in range(4):
        ax.annotate("", xy=(C1[i, 0], C1[i, 1]), xytext=(A[i, 0], A[i, 1]),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=2))
        ax.annotate("", xy=(C2[i, 0], C2[i, 1]), xytext=(C1[i, 0], C1[i, 1]),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$x_1$", fontsize=14)
    ax.set_ylabel(r"$x_2$", fontsize=14)
    ax.set_title("Morris Design with Strict OFAT", fontsize=14)

    # Standard OFAT
    C1 = A.copy()
    C2 = A.copy()
    for i in range(4):
        C1[i, 0] = A[i, 0] + delta
        if C1[i, 0] > 1:
            C1[i, 0] = A[i, 0] - delta
        C2[i, 1] = A[i, 1] + delta
        if C2[i, 1] > 1:
            C2[i, 1] = A[i, 1] - delta

    ax = axes[1]
    ax.scatter(A[:, 0], A[:, 1], s=150, color="black", marker="o", linewidths=2)
    ax.scatter(C1[:, 0], C1[:, 1], s=150, color="red", marker="^", linewidths=2)
    ax.scatter(C2[:, 0], C2[:, 1], s=150, color="green", marker="s", linewidths=2)
    for i in range(4):
        ax.annotate("", xy=(C1[i, 0], C1[i, 1]), xytext=(A[i, 0], A[i, 1]),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=2))
        ax.annotate("", xy=(C2[i, 0], C2[i, 1]), xytext=(A[i, 0], A[i, 1]),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$x_1$", fontsize=14)
    ax.set_ylabel(r"$x_2$", fontsize=14)
    ax.set_title("Morris Design with Standard OFAT", fontsize=14)

    fig.tight_layout()
    paths.append(_save(fig, out_dir / "morris_design.png"))

    # ------------------------------------------------------------------
    # Figure 6.5: Morris mu/sigma plot + boxplot
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch6_morris.R",
            inputs={"params.csv": pd.DataFrame({"p": [8], "r": [4], "n_rep": [100], "seed": [1]})},
            args=[],
            required_packages=["sensitivity"],
        )
        morris_single = out["morris_single.csv"]
        morris_boxplot = out["morris_boxplot.csv"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Mu-sigma plot
        ax = axes[0]
        mu = morris_single["mu"].values
        sigma = morris_single["sigma"].values
        variables = morris_single["variable"].values

        # Highlight important variables (mu > threshold)
        threshold = 10
        important = mu > threshold

        ax.scatter(mu[~important], sigma[~important], s=80, color="gray", alpha=0.6)
        for i in np.where(important)[0]:
            ax.annotate(variables[i], (mu[i], sigma[i]), fontsize=14, color="red",
                        ha="center", va="bottom")
        ax.set_xlabel(r"$|\mu|$", fontsize=14)
        ax.set_ylabel(r"$\sigma$", fontsize=14)
        ax.set_title("Borehole function", fontsize=14)

        # Boxplot of mu*
        ax = axes[1]
        data = [morris_boxplot[f"x{j}"].values for j in range(1, 9)]
        bp = ax.boxplot(data, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
        ax.set_xticklabels([f"x{j}" for j in range(1, 9)], fontsize=12)
        ax.set_ylabel(r"$\mu^*$", fontsize=14)
        ax.set_title("Morris screening design", fontsize=14)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "morris_analysis.png"))

    # ------------------------------------------------------------------
    # Figure 6.6: Sobol design visualization (pure Python)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, seed in zip(axes, [3, 1]):
        rng = np.random.default_rng(seed)
        A = rng.random((4, 2))
        B = rng.random((4, 2))
        C1 = np.column_stack([B[:, 0], A[:, 1]])
        C2 = np.column_stack([A[:, 0], B[:, 1]])

        ax.scatter(A[:, 0], A[:, 1], s=150, color="black", marker="o", linewidths=2)
        ax.scatter(C1[:, 0], C1[:, 1], s=150, color="red", marker="^", linewidths=2)
        ax.scatter(C2[:, 0], C2[:, 1], s=150, color="green", marker="s", linewidths=2)

        for i in range(4):
            ax.annotate("", xy=(C1[i, 0], C1[i, 1]), xytext=(A[i, 0], A[i, 1]),
                        arrowprops=dict(arrowstyle="->", color="blue", lw=2))
            ax.annotate("", xy=(C2[i, 0], C2[i, 1]), xytext=(A[i, 0], A[i, 1]),
                        arrowprops=dict(arrowstyle="->", color="blue", lw=2))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$x_1$", fontsize=14)
        ax.set_ylabel(r"$x_2$", fontsize=14)
        ax.set_title("Sobol' Design", fontsize=14)

    fig.tight_layout()
    paths.append(_save(fig, out_dir / "sobol_design.png"))

    # ------------------------------------------------------------------
    # Figure 6.7: MOFAT design visualization
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch6_mofat.R",
            inputs={"params.csv": pd.DataFrame({"p": [2], "l": [4], "n_rep": [100], "seed": [1]})},
            args=[],
            required_packages=["MOFAT"],
        )
        D_uniform = out["mofat_uniform_2d.csv"].to_numpy()
        D_projection = out["mofat_projection_2d.csv"].to_numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, D, title in zip(axes, [D_uniform, D_projection],
                                 ["MOFAT Design", "MOFAT Design with projections"]):
            l = 4
            A = D[:l, :]
            C1 = D[l:2 * l, :]
            C2 = D[2 * l:3 * l, :]

            ax.scatter(A[:, 0], A[:, 1], s=150, color="black", marker="o", linewidths=2)
            ax.scatter(C1[:, 0], C1[:, 1], s=150, color="red", marker="^", linewidths=2)
            ax.scatter(C2[:, 0], C2[:, 1], s=150, color="green", marker="s", linewidths=2)

            for i in range(l):
                ax.annotate("", xy=(C1[i, 0], C1[i, 1]), xytext=(A[i, 0], A[i, 1]),
                            arrowprops=dict(arrowstyle="->", color="blue", lw=2))
                ax.annotate("", xy=(C2[i, 0], C2[i, 1]), xytext=(A[i, 0], A[i, 1]),
                            arrowprops=dict(arrowstyle="->", color="blue", lw=2))

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel(r"$x_1$", fontsize=14)
            ax.set_ylabel(r"$x_2$", fontsize=14)
            ax.set_title(title, fontsize=14)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "mofat_design.png"))

    # ------------------------------------------------------------------
    # Figure 6.8: MOFAT borehole boxplot
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch6_mofat.R",
            inputs={"params.csv": pd.DataFrame({"p": [8], "l": [4], "n_rep": [100], "seed": [1]})},
            args=[],
            required_packages=["MOFAT"],
        )
        mofat_boxplot = out["mofat_boxplot.csv"]

        fig, ax = plt.subplots(figsize=(8, 6))
        data = [mofat_boxplot[f"x{j}"].values for j in range(1, 9)]
        bp = ax.boxplot(data, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
        ax.set_xticklabels([f"x{j}" for j in range(1, 9)], fontsize=12)
        ax.set_ylabel(r"$\mu^*$", fontsize=14)
        ax.set_title("MOFAT design", fontsize=14)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "mofat_borehole.png"))

    return out_dir, paths


def main(argv: Sequence[str] | None = None) -> int:
    generate_figures(use_r=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

