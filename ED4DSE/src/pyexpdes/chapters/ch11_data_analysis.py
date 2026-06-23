"""
Python port of `R files/Ch11.R` (Data Analysis).

Sections covered:
- 11.1 Factor Selection and Ranking (FIRST algorithm, RF/GP comparison)
- 11.2 Twin-Gaussian Process (TwinGP)

Python-first policy:
- Plotting is done in Python (matplotlib).
- R is invoked for: first (FIRST algorithm), randomForest, rkriging, twingp.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyexpdes.core.paths import figures_dir, datasets_dir
from pyexpdes.core.style import apply_base_style
from pyexpdes.core.r_utils import run_wrapper


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure Generation
# ---------------------------------------------------------------------------

def generate_figures(*, use_r: bool = True) -> tuple[Path, list[Path]]:
    """
    Generate Chapter 11 figures.

    Returns (output_dir, [figure_paths...]).
    """
    apply_base_style()
    out_dir = figures_dir("ch11")
    paths: list[Path] = []

    # ------------------------------------------------------------------
    # Figure 11.1: Effects and Total Sobol' Variance
    # ------------------------------------------------------------------
    # This is pure Python - theoretical curves showing effects and total variance

    x = np.linspace(0, 1, 100)
    rho_vals = np.linspace(-1, 1, 100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Effects
    ax = axes[0]
    ax.plot(x, np.sqrt(2) * x, color="red", linewidth=3, label=r"$x_1$")
    ax.plot(x, 2 / np.sqrt(3) * x, color="green", linewidth=3, label=r"$x_2$")
    ax.plot(x, 1 * x, color="blue", linewidth=3, label=r"$x_3$")
    ax.text(0.8, 1.22, r"$x_1$", fontsize=14, color="red")
    ax.text(0.8, 1.0, r"$x_2$", fontsize=14, color="green")
    ax.text(0.8, 0.7, r"$x_3$", fontsize=14, color="blue")
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    ax.set_title("Effects", fontsize=16)

    # Right: Total Sobol' Variance
    ax = axes[1]
    ax.plot(rho_vals, 2 * (1 - rho_vals**2), color="red", linewidth=3, label=r"$x_1$")
    ax.plot(rho_vals, 4/3 * (1 - rho_vals**2), color="green", linewidth=3, label=r"$x_2$")
    ax.axhline(y=1, color="blue", linewidth=3, label=r"$x_3$")
    ax.text(0, 1.9, r"$x_1$", fontsize=14, color="red")
    ax.text(0, 1.4, r"$x_2$", fontsize=14, color="green")
    ax.text(0, 0.9, r"$x_3$", fontsize=14, color="blue")
    ax.set_xlabel(r"$\rho$", fontsize=14)
    ax.set_ylabel(r"$V^{tot}$", fontsize=14)
    ax.set_title("Total Sobol'", fontsize=16)

    fig.tight_layout()
    paths.append(_save(fig, out_dir / "fig11_1_effects_sobol.png"))

    # ------------------------------------------------------------------
    # FIRST Algorithm Results (synthetic correlated data example)
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch11_first_example.R",
            inputs={
                "params.csv": pd.DataFrame({
                    "seed": [123],
                    "N": [1000],
                    "rho": [0.9],  # Correlation between x1 and x2
                }),
            },
            args=[],
            required_packages=["MASS", "first"],
        )

        first_scores = out["first_scores.csv"]
        first_without_x2 = out["first_scores_without_x2.csv"]
        first_rankings = out["first_rankings.csv"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: FIRST importance scores (all variables)
        ax = axes[0]
        colors = ["red", "green", "blue"]
        bars = ax.bar(first_scores["variable"], first_scores["importance"], color=colors)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xlabel("Variable", fontsize=14)
        ax.set_ylabel("FIRST Importance", fontsize=14)
        ax.set_title(r"FIRST Algorithm ($\rho=0.9$)", fontsize=16)

        # Annotate values
        for bar, val in zip(bars, first_scores["importance"]):
            height = bar.get_height()
            ax.annotate(f"{val:.3f}",
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       ha="center", va="bottom" if height >= 0 else "top",
                       fontsize=12)

        # Right: Comparison with/without x2
        ax = axes[1]
        x = np.arange(2)
        width = 0.35

        # Full model importance for x1, x3
        full_vals = [first_scores[first_scores["variable"] == "x1"]["importance"].values[0],
                     first_scores[first_scores["variable"] == "x3"]["importance"].values[0]]
        without_x2_vals = first_without_x2["importance"].values

        bars1 = ax.bar(x - width/2, full_vals, width, label="Full model", color="steelblue")
        bars2 = ax.bar(x + width/2, without_x2_vals, width, label="Without $x_2$", color="coral")

        ax.set_ylabel("FIRST Importance", fontsize=14)
        ax.set_title("Effect of Removing Correlated Variable", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(["$x_1$", "$x_3$"])
        ax.legend(fontsize=12)
        ax.axhline(y=0, color="black", linewidth=0.5)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "fig11_1b_first_results.png"))

    # ------------------------------------------------------------------
    # Figure 11.2: Factor Selection - RF vs GP Comparison
    # ------------------------------------------------------------------
    if use_r:
        concrete = pd.read_csv(datasets_dir() / "Concrete_Data.csv")

        out = run_wrapper(
            "ch11_factor_comparison.R",
            inputs={
                "params.csv": pd.DataFrame({
                    "seed": [123],
                    "n_rep": [30],
                    "split_ratio": [0.1],
                }),
                "concrete.csv": concrete,
            },
            args=[],
            required_packages=["SPlit", "randomForest", "rkriging", "first"],
        )

        rmse_rf = out["rmse_rf.csv"]
        rmse_gp = out["rmse_gp.csv"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Random Forest
        ax = axes[0]
        bp = ax.boxplot([rmse_rf["full"], rmse_rf["select"]],
                        labels=["full", "select"], patch_artist=True)
        bp["boxes"][0].set_facecolor("red")
        bp["boxes"][1].set_facecolor("green")
        ax.set_ylim(0, 0.37)
        ax.set_ylabel("RMSE", fontsize=14)
        ax.set_title("Random Forest", fontsize=16)

        # Right: Gaussian Process
        ax = axes[1]
        bp = ax.boxplot([rmse_gp["full"], rmse_gp["select"]],
                        labels=["full", "select"], patch_artist=True)
        bp["boxes"][0].set_facecolor("red")
        bp["boxes"][1].set_facecolor("green")
        ax.set_ylim(0, 0.37)
        ax.set_ylabel("RMSE", fontsize=14)
        ax.set_title("Gaussian process", fontsize=16)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "fig11_2_factor_comparison.png"))

    # ------------------------------------------------------------------
    # Figure 11.3: TwinGP
    # ------------------------------------------------------------------
    if use_r:
        wind = pd.read_csv(datasets_dir() / "wind10.csv")

        out = run_wrapper(
            "ch11_twingp.R",
            inputs={
                "params.csv": pd.DataFrame({
                    "seed": [123],
                }),
                "wind.csv": wind,
            },
            args=[],
            required_packages=["twingp"],
        )

        D = wind[["WindSpeed", "Power"]].values
        pred = out["prediction.csv"]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(D[:, 0], D[:, 1], marker="x", s=15, linewidths=0.5, color="gray")
        ax.plot(pred["speed"].values, pred["power"].values, color="blue", linewidth=3)
        ax.set_xlabel("speed", fontsize=14)
        ax.set_ylabel("power", fontsize=14)
        ax.set_title("TwinGP", fontsize=16)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "fig11_3_twingp.png"))

    return out_dir, paths


def main(argv: Sequence[str] | None = None) -> int:
    generate_figures(use_r=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
