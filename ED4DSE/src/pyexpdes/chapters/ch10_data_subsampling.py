"""
Python port of `R files/Ch10.R` (Data Subsampling).

Sections covered:
- 10.1 Support Points-based Subsampling (sp, subsample)
- 10.2 Data Splitting (SPlit)
- 10.3 Data Twinning (twin, energy, spatial balance)
- 10.4 Supervised Compression (supercompress)

Python-first policy:
- Plotting is done in Python (matplotlib).
- R is invoked for: support (sp, subsample), SPlit, twinning (twin),
  BalancedSampling (lpm1, lpm2, energy, sb), supercompress, deldir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

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
    Generate Chapter 10 figures.

    Returns (output_dir, [figure_paths...]).
    """
    apply_base_style()
    out_dir = figures_dir("ch10")
    paths: list[Path] = []

    # ------------------------------------------------------------------
    # Figure 10.1: Support Points vs Subsample
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch10_support_subsample.R",
            inputs={
                "params.csv": pd.DataFrame({
                    "seed": [123],
                    "N": [1000],
                    "n": [100],
                    "p": [2],
                    "rho": [0.5],
                })
            },
            args=[],
            required_packages=["support", "MASS"],
        )

        D = out["data.csv"].values
        S = out["support_points.csv"].values
        subsamp = out["subsample.csv"].values

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Support Points
        ax = axes[0]
        ax.scatter(D[:, 0], D[:, 1], marker="x", s=20, linewidths=0.5, color="gray")
        ax.scatter(S[:, 0], S[:, 1], s=80, facecolors="none", edgecolors="red", linewidths=2)
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.set_title("Support Points", fontsize=16)

        # Right: Subsample
        ax = axes[1]
        ax.scatter(D[:, 0], D[:, 1], marker="x", s=20, linewidths=0.5, color="gray")
        ax.scatter(subsamp[:, 0], subsamp[:, 1], s=80, facecolors="none",
                   edgecolors="red", linewidths=2)
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.set_title("Subsample", fontsize=16)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "fig10_1_support_subsample.png"))

    # ------------------------------------------------------------------
    # Figure 10.2: Wind Energy Data with GP fit
    # ------------------------------------------------------------------
    if use_r:
        wind_data = pd.read_csv(datasets_dir() / "wind10.csv")

        out = run_wrapper(
            "ch10_wind_gp.R",
            inputs={
                "params.csv": pd.DataFrame({
                    "seed": [123],
                    "n": [392],
                }),
                "wind.csv": wind_data,
            },
            args=[],
            required_packages=["support", "rkriging"],
        )

        D = wind_data[["WindSpeed", "Power"]].values
        S = out["subsample.csv"].values
        pred = out["prediction.csv"]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(D[:, 0], D[:, 1], marker="x", s=15, linewidths=0.5, color="gray",
                   label="data")
        ax.scatter(S[:, 0], S[:, 1], s=50, facecolors="none", edgecolors="red",
                   linewidths=1.5, label="subsample")
        ax.plot(pred["speed"].values, pred["power"].values, color="blue", linewidth=2, label="GP")
        ax.set_xlabel("speed", fontsize=14)
        ax.set_ylabel("power", fontsize=14)
        ax.set_title("wind power data", fontsize=16)
        ax.legend(loc="lower right", fontsize=12, frameon=False)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "fig10_2_wind_gp.png"))

    # ------------------------------------------------------------------
    # Figure 10.3: Random vs SPlit for Data Splitting
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch10_split.R",
            inputs={
                "params.csv": pd.DataFrame({
                    "seed": [102],
                    "N": [100],
                    "Ntest": [20],
                })
            },
            args=[],
            required_packages=["SPlit"],
        )

        D = out["data.csv"].values
        y = out["labels.csv"]["y"].values.astype(int)
        random_idx = out["random_idx.csv"]["idx"].values.astype(int) - 1  # R is 1-indexed
        split_idx = out["split_idx.csv"]["idx"].values.astype(int) - 1

        colors = ["red", "green", "blue"]
        markers = ["^", "s", "o"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Random
        ax = axes[0]
        for k in range(1, 4):
            mask = y == k
            ax.scatter(D[mask, 0], D[mask, 1], c=colors[k-1], marker=markers[k-1],
                       s=80, linewidths=1.5)
        ax.scatter(D[random_idx, 0], D[random_idx, 1], s=300, facecolors="none",
                   edgecolors="black", linewidths=2)
        ax.set_xlabel(r"$x_1$", fontsize=14)
        ax.set_ylabel(r"$x_2$", fontsize=14)
        ax.set_title("Random", fontsize=16)

        # Right: SPlit
        ax = axes[1]
        for k in range(1, 4):
            mask = y == k
            ax.scatter(D[mask, 0], D[mask, 1], c=colors[k-1], marker=markers[k-1],
                       s=80, linewidths=1.5)
        ax.scatter(D[split_idx, 0], D[split_idx, 1], s=300, facecolors="none",
                   edgecolors="black", linewidths=2)
        ax.set_xlabel(r"$x_1$", fontsize=14)
        ax.set_ylabel(r"$x_2$", fontsize=14)
        ax.set_title("SPlit", fontsize=16)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "fig10_3_split.png"))

    # ------------------------------------------------------------------
    # Figure 10.4: Twinning Example
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch10_twinning.R",
            inputs={
                "params.csv": pd.DataFrame({
                    "seed": [6],
                    "N": [50],
                    "r": [5],  # N/n ratio
                })
            },
            args=[],
            required_packages=["twinning"],
        )

        D = out["data.csv"].values
        twin_idx = out["twin_idx.csv"]["idx"].values.astype(int) - 1

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(D[:, 0], D[:, 1], s=30, color="black")
        ax.scatter(D[twin_idx, 0], D[twin_idx, 1], s=100, facecolors="none",
                   edgecolors="red", linewidths=2)
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.set_title("Data Twinning", fontsize=16)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "fig10_4_twinning.png"))

    # ------------------------------------------------------------------
    # Figure 10.5: Sampling Methods Comparison (Energy Distance & Spatial Balance)
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch10_sampling_comparison.R",
            inputs={
                "params.csv": pd.DataFrame({
                    "seed": [4],
                    "N": [1000],
                    "n": [100],
                    "n_rep": [30],
                })
            },
            args=[],
            required_packages=["twinning", "SPlit", "BalancedSampling"],
        )

        ed = out["energy_distance.csv"]
        sb = out["spatial_balance.csv"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Energy Distance
        ax = axes[0]
        data = [ed["SPlit"], ed["twin"], ed["LPM1"], ed["LPM2"]]
        bp = ax.boxplot(data, labels=["SPlit", "twin", "LPM1", "LPM2"], patch_artist=True)
        colors_box = ["yellow", "red", "green", "blue"]
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
        ax.set_xlabel("sampling methods", fontsize=14)
        ax.set_ylabel("energy distance", fontsize=14)
        ax.set_title("Energy Distance", fontsize=16)

        # Right: Spatial Balance
        ax = axes[1]
        data = [sb["SPlit"], sb["twin"], sb["LPM1"], sb["LPM2"]]
        bp = ax.boxplot(data, labels=["SPlit", "twin", "LPM1", "LPM2"], patch_artist=True)
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
        ax.set_xlabel("sampling methods", fontsize=14)
        ax.set_ylabel("spatial balance measure", fontsize=14)
        ax.set_title("Spatial Balance", fontsize=16)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "fig10_5_sampling_comparison.png"))

    # ------------------------------------------------------------------
    # Figure 10.6: Supervised vs Unsupervised Compression
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch10_supercompress.R",
            inputs={
                "params.csv": pd.DataFrame({
                    "seed": [1],
                    "N": [20000],  # 10000*p
                    "n": [200],
                    "p": [2],
                })
            },
            args=[],
            required_packages=["supercompress", "FNN"],
        )

        D_kmeans = out["kmeans_centers.csv"].values
        D_super = out["supercompress_centers.csv"].values
        grid = out["grid.csv"]

        # Reshape grid for contour plot
        N_plot = 250
        p1 = np.linspace(0, 1, N_plot)
        p2 = np.linspace(0, 1, N_plot)
        fc = grid["f"].values.reshape(N_plot, N_plot)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Unsupervised (K-means)
        ax = axes[0]
        cf = ax.contourf(p1, p2, fc.T, levels=20, cmap="coolwarm")
        ax.scatter(D_kmeans[:, 0], D_kmeans[:, 1], s=40, color="black")

        # Voronoi tessellation
        vor = Voronoi(D_kmeans)
        for simplex in vor.ridge_vertices:
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                ax.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], "k-", linewidth=0.5)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$x_1$", fontsize=14)
        ax.set_ylabel(r"$x_2$", fontsize=14)
        ax.set_title("unsupervised", fontsize=16)

        # Right: Supervised (supercompress)
        ax = axes[1]
        cf = ax.contourf(p1, p2, fc.T, levels=20, cmap="coolwarm")
        ax.scatter(D_super[:, 0], D_super[:, 1], s=40, color="black")

        vor = Voronoi(D_super)
        for simplex in vor.ridge_vertices:
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                ax.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], "k-", linewidth=0.5)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$x_1$", fontsize=14)
        ax.set_ylabel(r"$x_2$", fontsize=14)
        ax.set_title("supervised", fontsize=16)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "fig10_6_supercompress.png"))

    return out_dir, paths


def main(argv: Sequence[str] | None = None) -> int:
    generate_figures(use_r=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
