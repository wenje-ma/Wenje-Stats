"""
Python port of `R files/Ch7.R` (Sequential Designs).

Sections covered:
- 7.1 Emulation (ALC, ALMV, ALM - prediction-based sequential design)
- 7.2 Optimization (Expected Improvement)
- 7.3 Inverse designs (OSFD)

Python-first policy:
- Plotting is done in Python (matplotlib).
- R is invoked for: rkriging, MaxPro, spacefillr, OSFD.
- Sequential algorithms (ALC, ALMV, ALM) are implemented in R due to complex kriging operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from pyexpdes.core.paths import figures_dir
from pyexpdes.core.style import apply_base_style
from pyexpdes.core.r_utils import run_wrapper


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _f_1d(x: float) -> float:
    """1D test function for EI example."""
    return np.sin(10 * np.pi * x) / (1 + 64 * (x - 0.25) ** 2) + x**2


def _plot_sequential_design(D: np.ndarray, rmse: np.ndarray, n0: int, title: str) -> plt.Figure:
    """
    Plot sequential design results in 3-panel layout.

    Args:
        D: Design matrix (n0 + n_new) x p
        rmse: RMSE history array
        n0: Number of initial design points
        title: Plot title (method name)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    n_new = D.shape[0] - n0

    # Plot 1: x1 vs x2
    ax = axes[0]
    ax.scatter(D[:n0, 0], D[:n0, 1], s=150, color="blue", label="initial")
    for i in range(n_new):
        ax.annotate(
            str(i + 1),
            (D[n0 + i, 0], D[n0 + i, 1]),
            fontsize=14,
            color="red",
            ha="center",
            va="center",
            fontweight="bold",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$x_1$", fontsize=14)
    ax.set_ylabel(r"$x_2$", fontsize=14)

    # Plot 2: x3 vs x4
    ax = axes[1]
    ax.scatter(D[:n0, 2], D[:n0, 3], s=150, color="blue", label="initial")
    for i in range(n_new):
        ax.annotate(
            str(i + 1),
            (D[n0 + i, 2], D[n0 + i, 3]),
            fontsize=14,
            color="red",
            ha="center",
            va="center",
            fontweight="bold",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$x_3$", fontsize=14)
    ax.set_ylabel(r"$x_4$", fontsize=14)

    # Plot 3: RMSE curve
    ax = axes[2]
    ax.plot(range(len(rmse)), rmse, marker="x", markersize=12, linewidth=2, color="blue", mew=3)
    ax.set_xlabel("new design run", fontsize=14)
    ax.set_ylabel("RMSE", fontsize=14)
    ax.set_xticks(range(len(rmse)))
    ax.set_xticklabels(range(len(rmse)))

    fig.suptitle(title, fontsize=18)
    fig.tight_layout()
    return fig


def generate_figures(*, use_r: bool = True) -> tuple[Path, list[Path]]:
    """
    Generate Chapter 7 figures.

    Returns (output_dir, [figure_paths...]).
    """
    apply_base_style()
    out_dir = figures_dir("ch07")
    paths: list[Path] = []

    # Sequential design parameters
    p = 4
    n0 = 20
    n_new = 10

    # ------------------------------------------------------------------
    # Figure 7.2: ALC (Active Learning Cohn)
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch7_sequential_design.R",
            inputs={
                "params.csv": pd.DataFrame({
                    "method": ["ALC"],
                    "p": [p],
                    "n0": [n0],
                    "n_new": [n_new],
                    "seed": [1],
                })
            },
            args=[],
            required_packages=["rkriging", "MaxPro", "spacefillr", "FNN", "pdist"],
        )
        D = out["design.csv"].to_numpy()
        rmse = out["rmse.csv"]["rmse"].to_numpy()

        fig = _plot_sequential_design(D, rmse, n0, "ALC (Active Learning Cohn)")
        paths.append(_save(fig, out_dir / "sequential_alc.png"))

    # ------------------------------------------------------------------
    # Figure 7.3: ALMV (Active Learning MacKay Variance)
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch7_sequential_design.R",
            inputs={
                "params.csv": pd.DataFrame({
                    "method": ["ALMV"],
                    "p": [p],
                    "n0": [n0],
                    "n_new": [n_new],
                    "seed": [1],
                })
            },
            args=[],
            required_packages=["rkriging", "MaxPro", "spacefillr", "FNN", "pdist"],
        )
        D = out["design.csv"].to_numpy()
        rmse = out["rmse.csv"]["rmse"].to_numpy()

        fig = _plot_sequential_design(D, rmse, n0, "ALMV (Active Learning MacKay Variance)")
        paths.append(_save(fig, out_dir / "sequential_almv.png"))

    # ------------------------------------------------------------------
    # Figure 7.4: ALM (Active Learning MacKay)
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch7_sequential_design.R",
            inputs={
                "params.csv": pd.DataFrame({
                    "method": ["ALM"],
                    "p": [p],
                    "n0": [n0],
                    "n_new": [n_new],
                    "seed": [1],
                })
            },
            args=[],
            required_packages=["rkriging", "MaxPro", "spacefillr", "FNN", "pdist"],
        )
        D = out["design.csv"].to_numpy()
        rmse = out["rmse.csv"]["rmse"].to_numpy()

        fig = _plot_sequential_design(D, rmse, n0, "ALM (Active Learning MacKay)")
        paths.append(_save(fig, out_dir / "sequential_alm.png"))

    # ------------------------------------------------------------------
    # Figure 7.5: Expected Improvement for optimization
    # ------------------------------------------------------------------
    if use_r:
        # Use R for kriging in EI as well
        n = 8
        D = ((np.arange(1, n + 1) - 0.5) / n).reshape(-1, 1)
        y = np.array([_f_1d(x[0]) for x in D])
        test = np.linspace(0, 1, 301)
        true_y = np.array([_f_1d(x) for x in test])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        titles = ["Initial", "Step 1", "Step 2"]

        for step, ax in enumerate(axes):
            # Fit kriging model via R
            out = run_wrapper(
                "ch7_kriging_fit_predict.R",
                inputs={
                    "design.csv": pd.DataFrame(D),
                    "response.csv": pd.DataFrame({"y": y}),
                    "test.csv": pd.DataFrame({"x": test}),
                },
                args=[],
                required_packages=["rkriging"],
            )
            pred_mean = out["predictions.csv"]["mean"].to_numpy()
            pred_sd = out["predictions.csv"]["sd"].to_numpy()

            # Plot
            ax.plot(test, true_y, color="green", linewidth=3, linestyle=":", label="truth")
            ax.fill_between(
                test,
                pred_mean - 2 * pred_sd,
                pred_mean + 2 * pred_sd,
                color="lightgray",
                alpha=0.8,
            )
            ax.plot(test, pred_mean, color="black", linewidth=2, linestyle="--", label="prediction")
            ax.scatter(D.flatten(), y, s=150, color="blue", zorder=5)

            # Mark new points from previous steps
            if step > 0:
                for i in range(n, n + step):
                    ax.annotate(
                        str(i + 1),
                        (D[i, 0], y[i]),
                        fontsize=14,
                        color="blue",
                        ha="center",
                        va="bottom",
                    )

            # EI curve
            y_min = y.min()
            ei = np.zeros(len(test))
            for i, x_test in enumerate(test):
                if pred_sd[i] > 1e-10:
                    z = (y_min - pred_mean[i]) / pred_sd[i]
                    ei[i] = pred_sd[i] * (z * norm.cdf(z) + norm.pdf(z))

            ax2 = ax.twinx()
            ax2.plot(test, ei, color="red", linewidth=2, label="EI")
            ax2.set_ylabel("EI", fontsize=14, color="red")
            ax2.tick_params(axis="y", labelcolor="red")

            # Mark next point
            x_next_idx = np.argmax(ei)
            x_next = test[x_next_idx]
            ax.scatter([x_next], [0], s=200, color="red", marker="*", zorder=6)

            ax.set_xlabel("x", fontsize=14)
            ax.set_ylabel("y", fontsize=14)
            ax.set_title(titles[step], fontsize=16)
            ax.set_ylim(min(true_y) - 0.3, max(true_y) + 0.5)
            ax.legend(["truth", "prediction", "EI"], loc="upper left", frameon=False)

            if step < 2:
                # Add next point for next iteration
                D = np.vstack([D, [[x_next]]])
                y = np.append(y, _f_1d(x_next))

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "expected_improvement.png"))

    # ------------------------------------------------------------------
    # Figure 7.6: OSFD (Output Space Filling Design)
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch7_osfd.R",
            inputs={
                "params.csv": pd.DataFrame({
                    "p": [2],
                    "q": [2],
                    "n": [50],
                    "n0": [10],
                    "seed": [1],
                })
            },
            args=[],
            required_packages=["OSFD", "MaxPro"],
        )
        D = out["design_D.csv"].to_numpy()
        Y = out["design_Y.csv"].to_numpy()
        n0_osfd = 10

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Input space
        ax = axes[0]
        ax.scatter(D[:n0_osfd, 0], D[:n0_osfd, 1], s=100, color="blue", label="initial")
        ax.scatter(
            D[n0_osfd:, 0],
            D[n0_osfd:, 1],
            s=100,
            color="red",
            marker="x",
            linewidths=2,
            label="sequential",
        )
        ax.set_xlabel(r"$x_1$", fontsize=14)
        ax.set_ylabel(r"$x_2$", fontsize=14)
        ax.set_title("input space", fontsize=16)
        ax.legend(loc="upper right", frameon=False)

        # Output space
        ax = axes[1]
        ax.scatter(Y[:n0_osfd, 0], Y[:n0_osfd, 1], s=100, color="blue", label="initial")
        ax.scatter(
            Y[n0_osfd:, 0],
            Y[n0_osfd:, 1],
            s=100,
            color="red",
            marker="x",
            linewidths=2,
            label="sequential",
        )
        ax.set_xlabel(r"$y_1$", fontsize=14)
        ax.set_ylabel(r"$y_2$", fontsize=14)
        ax.set_title("output space", fontsize=16)
        ax.legend(loc="upper right", frameon=False)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "osfd.png"))

    return out_dir, paths


def main(argv: Sequence[str] | None = None) -> int:
    generate_figures(use_r=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
