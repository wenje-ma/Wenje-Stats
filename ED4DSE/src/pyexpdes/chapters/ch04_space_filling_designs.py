"""
Python port of `R files/ch4.R` - Space-Filling Designs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.spatial import Voronoi, voronoi_plot_2d

from pyexpdes.core.paths import figures_dir
from pyexpdes.core.style import apply_base_style
from pyexpdes.core.r_utils import run_wrapper





def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _lattice(n: int, p: int) -> np.ndarray:
    # Simple lattice candidate set in [0,1]^p
    # (Not identical to LatticeDesign::Lattice, but deterministic and dense.)
    grid_1d = (np.arange(n) + 0.5) / n
    if p == 2:
        X, Y = np.meshgrid(grid_1d, grid_1d, indexing="xy")
        pts = np.column_stack([X.ravel(), Y.ravel()])
        return pts
    raise NotImplementedError("Only p=2 lattice is used in this chapter subset.")


def _kmeans_pp(X: np.ndarray, k: int, seed: int = 2, iters: int = 50) -> np.ndarray:
    """
    Minimal k-means for deterministic clustering centers.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=k, replace=False)
    C = X[idx].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        lab = d.argmin(axis=1)
        C_new = np.vstack([X[lab == j].mean(axis=0) for j in range(k)])
        if np.allclose(C_new, C):
            break
        C = C_new
    return C


def _draw_grid_lines(ax: plt.Axes, D: np.ndarray) -> None:
    """
    Match the grid-line drawing style in ch4.R where they compute midpoints.
    """
    n = D.shape[0]
    d1 = np.sort(D[:, 0])
    d2 = np.sort(D[:, 1])
    l1 = np.concatenate([[0.0], (d1[1:] + d1[:-1]) / 2.0, [1.0]])
    l2 = np.concatenate([[0.0], (d2[1:] + d2[:-1]) / 2.0, [1.0]])
    for x in l1[1:-1]:
        ax.plot([x, x], [0, 1], color="black", linewidth=1)
    for y in l2[1:-1]:
        ax.plot([0, 1], [y, y], color="black", linewidth=1)


def _set_axes_like_r(ax: plt.Axes, *, xlim=(-0.1, 1.1), ylim=(-0.1, 1.1)) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")


def _draw_circles(ax: plt.Axes, D: np.ndarray, r: float, *, color="gray", lw: float = 2.0) -> None:
    for (x0, y0) in D:
        ax.add_artist(plt.Circle((x0, y0), r, fill=False, color=color, linewidth=lw))


def _lhs_random(n: int, p: int, seed: int = 1) -> np.ndarray:
    """
    Simple random Latin hypercube in [0,1]^p.
    (Used for constrained candidate generation; matches lhs::randomLHS intent.)
    """
    rng = np.random.default_rng(seed)
    H = np.empty((n, p), float)
    for j in range(p):
        perm = rng.permutation(n)
        u = rng.random(n)
        H[:, j] = (perm + u) / n
    return H


def _constraint_vals(x: np.ndarray) -> np.ndarray:
    x1, x2 = float(x[0]), float(x[1])
    c1 = (x1 - np.sqrt(50 * (x2 - 0.52) ** 2 + 2) + 1)
    c2 = (np.sqrt(120 * (x2 - 0.48) ** 2 + 1) - 0.75 - x1)
    c3 = (0.65**2 - x1**2 - x2**2)
    return np.array([c1, c2, c3], float)


def _constraint_contour(num: int = 1001) -> tuple[np.ndarray, np.ndarray]:
    x2_seq = np.linspace(0, 1, num)
    x1_1 = np.sqrt(50 * (x2_seq - 0.52) ** 2 + 2) - 1
    x1_2 = np.sqrt(120 * (x2_seq - 0.48) ** 2 + 1) - 0.75
    x1_3 = np.sqrt(np.maximum(0.65**2 - x2_seq**2, 0.0))
    x1 = np.vstack([x1_1, x1_2, x1_3])
    x2 = np.vstack([x2_seq, x2_seq, x2_seq])
    return x1, x2


def _branin(u: np.ndarray) -> float:
    x1 = u[0] * 15 - 5
    x2 = u[1] * 15
    val1 = (x2 - 5.1 / (4 * np.pi**2) * (x1**2) + 5 / np.pi * x1 - 6) ** 2
    val2 = 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return float(val1 + val2)


def _scatter_matrix(D: np.ndarray, *, var_labels: list[str] | None = None, bins: int = 15) -> plt.Figure:
    """
    Minimal scatterplot matrix similar in spirit to car::scatterplotMatrix.
    """
    n, p = D.shape
    if var_labels is None:
        var_labels = [f"x{i+1}" for i in range(p)]

    fig, axes = plt.subplots(p, p, figsize=(2.2 * p, 2.2 * p))
    for i in range(p):
        for j in range(p):
            ax = axes[i, j]
            if i == j:
                ax.hist(D[:, j], bins=bins, color="lightgray", edgecolor="black")
            else:
                ax.scatter(D[:, j], D[:, i], s=8, color="blue", alpha=0.7)
            if i == p - 1:
                ax.set_xlabel(var_labels[j])
            else:
                ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(var_labels[i])
            else:
                ax.set_yticks([])
    fig.tight_layout()
    return fig


def _min_mindist(M: np.ndarray) -> np.ndarray:
    """
    Match ch4.R's min_mindist() intent:
    for each projection dimension j=1..(p-1), compute the minimum over all j-dim projections
    of (minimum pairwise distance); also include full-dimensional min distance.
    """
    n, p = M.shape
    out = np.zeros(p, float)
    for j in range(1, p):
        mins = []
        for cols in itertools.combinations(range(p), j):
            mins.append(float(np.min(pdist(M[:, cols]))))
        out[j - 1] = float(np.min(mins))
    out[p - 1] = float(np.min(pdist(M)))
    return out


def generate_figures(*, use_r: bool = True) -> tuple[Path, list[Path]]:
    apply_base_style()
    out_dir = figures_dir("ch04")
    paths: list[Path] = []

    # ------------------------------------------------------------------
    # Clustering-based design (Voronoi tessellation)
    # ------------------------------------------------------------------
    p = 2
    n = 7
    X = _lattice(100, p)  # dense candidates
    D = _kmeans_pp(X, k=n, seed=2, iters=100)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(D[:, 0], D[:, 1], s=150, color="blue", zorder=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("clustering-based design", fontsize=16)
    ax.set_xlabel(r"$x_1$", fontsize=14)
    ax.set_ylabel(r"$x_2$", fontsize=14)
    try:
        vor = Voronoi(D)
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors="gray", line_width=2)
    except Exception:
        pass
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "clustering_based.png"))

    # ------------------------------------------------------------------
    # Minimax design (theoretical + optimized via minimaxdesign)
    # ------------------------------------------------------------------
    DmM = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.0],
            [0.5, 1.0],
            [1 / 3 - 1 / 12 * np.sqrt(7), 0.25],
            [1 / 3 - 1 / 12 * np.sqrt(7), 0.75],
            [2 / 3 + 1 / 12 * np.sqrt(7), 0.25],
            [2 / 3 + 1 / 12 * np.sqrt(7), 0.75],
        ]
    )
    r = 1 / 6 * (np.sqrt(7) - 1)

    D_opt = None
    if use_r:
        try:
            out = run_wrapper(
                "ch4_minimax.R",
                inputs={"params.csv": pd.DataFrame({"n": [7], "p": [2], "q": [10], "seed": [2]})},
                args=[],
                required_packages=["minimaxdesign"],
            )
            D_opt = out["minimax_design.csv"].to_numpy()
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(DmM[:, 0], DmM[:, 1], s=150, color="blue", zorder=5)
    if D_opt is not None:
        ax.scatter(D_opt[:, 0], D_opt[:, 1], s=100, marker="^", color="green", zorder=6)
    for (x0, y0) in DmM:
        ax.add_artist(plt.Circle((x0, y0), r, fill=False, color="gray", linewidth=2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title("minimax design", fontsize=16)
    ax.set_xlabel(r"$x_1$", fontsize=14)
    ax.set_ylabel(r"$x_2$", fontsize=14)
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "minimax.png"))

    # ------------------------------------------------------------------
    # Projection-based designs (illustrative, 1x2)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    D1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    D2 = np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]], float)

    for ax, D, title in zip(axes, [D1, D2], ["maximin design", "minimax design"]):
        ax.scatter(D[:, 0], D[:, 1], s=200, color="blue")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "projection_based.png"))

    # ------------------------------------------------------------------
    # Maximin + sequential maximin + MmLHD (via small R wrapper for book-faithful designs)
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch4_maximin_mmlhd_2d.R",
            inputs={"noop.csv": pd.DataFrame({"x": [0.0]})},
            args=[],
            required_packages=["SFDesign"],
        )

        # Maximin design with circles
        D_mm = out["maximin_7_2.csv"].to_numpy()
        r_mm = float(out["maximin_7_2_radius.csv"]["r"].iloc[0])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(D_mm[:, 0], D_mm[:, 1], s=150, color="blue", zorder=5)
        _draw_circles(ax, D_mm, r_mm, color="gray", lw=2.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title("maximin design", fontsize=16)
        ax.set_xlabel(r"$x_1$", fontsize=14)
        ax.set_ylabel(r"$x_2$", fontsize=14)
        fig.tight_layout()
        paths.append(_save(fig, out_dir / "maximin.png"))

        # Sequential maximin (n=7 then n=13; show new points)
        D7 = out["seq_maximin_7_2.csv"].to_numpy()
        D13 = out["seq_maximin_13_2.csv"].to_numpy()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.scatter(D7[:, 0], D7[:, 1], s=150, color="blue")
        _set_axes_like_r(ax, xlim=(0, 1), ylim=(0, 1))
        ax.set_title("sequential maximin (n=7)", fontsize=14)

        ax = axes[1]
        ax.scatter(D13[:7, 0], D13[:7, 1], s=150, color="blue")
        ax.scatter(
            D13[7:, 0],
            D13[7:, 1],
            s=180,
            facecolors="none",
            edgecolors="red",
            marker="x",
            linewidths=2.5,
        )
        _set_axes_like_r(ax, xlim=(0, 1), ylim=(0, 1))
        ax.set_title("sequential maximin (n=13)", fontsize=14)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "sequential_maximin.png"))

        # MmLHD n=4 illustration (compare to diagonal)
        D_mmlhd_4 = out["mmlhd_4_2.csv"].to_numpy()
        D_diag_4 = (np.column_stack([np.arange(1, 5), np.arange(1, 5)]) - 0.5) / 4.0
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, D in zip(axes, [D_mmlhd_4, D_diag_4]):
            ax.scatter(D[:, 0], D[:, 1], s=220, color="blue")
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            ax.set_xticks(np.linspace(0, 1, 5))
            ax.set_yticks(np.linspace(0, 1, 5))
            for i in range(5):
                t = i / 4
                ax.plot([t, t], [0, 1], color="black", linewidth=1)
                ax.plot([0, 1], [t, t], color="black", linewidth=1)
            ax.scatter(D[:, 0], np.full(D.shape[0], -0.15), s=220, color="red")
            ax.scatter(np.full(D.shape[0], -0.15), D[:, 1], s=220, color="red")
        fig.tight_layout()
        paths.append(_save(fig, out_dir / "mmlhd_n4.png"))

        # MmLHD n=7
        D_mmlhd_7 = out["mmlhd_7_2.csv"].to_numpy()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(D_mmlhd_7[:, 0], D_mmlhd_7[:, 1], s=200, color="blue")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("MmLHD", fontsize=16)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        for i in range(8):
            t = i / 7
            ax.plot([t, t], [0, 1], color="black", linewidth=1)
            ax.plot([0, 1], [t, t], color="black", linewidth=1)
        fig.tight_layout()
        paths.append(_save(fig, out_dir / "mmlhd_n7.png"))

        # MmLHD n=20 and Chebyshev-MmLHD n=20
        D_mmlhd_20 = out["mmlhd_20_2.csv"].to_numpy()
        D_cheb_20 = out["chebyshev_mmlhd_20_2.csv"].to_numpy()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, D, title in zip(axes, [D_mmlhd_20, D_cheb_20], ["MmLHD", "Chebyshev-MmLHD"]):
            ax.scatter(D[:, 0], D[:, 1], s=70, color="blue")
            _set_axes_like_r(ax)
            ax.set_title(title, fontsize=14)
            if title == "MmLHD":
                for i in range(21):
                    t = i / 20
                    ax.plot([t, t], [0, 1], color="black", linewidth=0.8)
                    ax.plot([0, 1], [t, t], color="black", linewidth=0.8)
            else:
                _draw_grid_lines(ax, D)
        fig.tight_layout()
        paths.append(_save(fig, out_dir / "mmlhd_n20.png"))

    # ------------------------------------------------------------------
    # MaxPro designs (2 panels) - design points via R wrapper, plot in Python
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch4_maxpro_2d.R",
            inputs={"noop.csv": pd.DataFrame({"x": [0.0]})},
            args=[],
            required_packages=["MaxPro", "SFDesign"],
        )
        D7 = out["maxpro_7_2.csv"].to_numpy()
        D20 = out["maxpro_20_2.csv"].to_numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.scatter(D7[:, 0], D7[:, 1], s=120, color="blue")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title("MaxPro(7,2)", fontsize=14)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        _draw_grid_lines(ax, D7)

        ax = axes[1]
        ax.scatter(D20[:, 0], D20[:, 1], s=80, color="blue")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title("MaxPro(20,2)", fontsize=14)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        _draw_grid_lines(ax, D20)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "maxpro.png"))

    # ------------------------------------------------------------------
    # High-dimensional MaxPro vs MmLHD (p=5,n=50): scatter matrix + projection-distance plot
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch4_highdim_maxpro.R",
            inputs={"noop.csv": pd.DataFrame({"x": [0.0]})},
            args=[],
            required_packages=["MaxPro", "SFDesign"],
        )
        D_maxpro = out["maxpro_50_5.csv"].to_numpy()
        D_mmlhd = out["mmlhd_50_5.csv"].to_numpy()

        fig = _scatter_matrix(D_maxpro, var_labels=[r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$", r"$x_5$"], bins=15)
        paths.append(_save(fig, out_dir / "maxpro_50_5_scatter_matrix.png"))

        d1 = _min_mindist(D_maxpro)
        d2 = _min_mindist(D_mmlhd)
        p = D_maxpro.shape[1]

        fig, ax = plt.subplots(figsize=(8, 6))
        xs = np.arange(1, p + 1)
        ax.plot(xs, d1, marker="o", color="blue", linewidth=3, label="MaxPro")
        ax.plot(xs, d2, marker="x", color="red", linewidth=3, linestyle="--", label="MmLHD")
        ax.set_xlim(1, p)
        ax.set_xticks(xs)
        ax.set_xlabel("projection dimension")
        ax.set_ylabel("minimum of minimum distances")
        ax.legend(loc="upper left", frameon=False)
        fig.tight_layout()
        paths.append(_save(fig, out_dir / "projection_min_mindist.png"))

    # ------------------------------------------------------------------
    # Pair-wise distance distribution (Maximin vs MaxPro + Beta fit)
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch4_distance_distribution.R",
            inputs={"noop.csv": pd.DataFrame({"x": [0.0]})},
            args=[],
            required_packages=["SFDesign", "fitdistrplus"],
        )
        d0 = out["dist_maximin.csv"]["d0"].to_numpy()
        d1 = out["dist_maxpro.csv"]["d1"].to_numpy()
        a = float(out["beta_fit.csv"]["a"].iloc[0])
        b = float(out["beta_fit.csv"]["b"].iloc[0])
        p = int(out["beta_fit.csv"]["p"].iloc[0])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(d0, bins=30, density=True, color=(1, 0, 0, 0.25), label="Maximin")
        ax.hist(d1, bins=30, density=True, color=(0, 0, 1, 0.25), label="MaxPro")
        xs = np.linspace(0, np.sqrt(p), 400)
        from scipy.stats import beta as _beta

        eps = 1e-9
        z = np.clip(xs / np.sqrt(p), eps, 1 - eps)
        pdf = _beta.pdf(z, a, b) / np.sqrt(p)
        ax.plot(xs, pdf, color="green", linewidth=3, label=f"Beta({a:.1f},{b:.1f})")
        ax.set_xlim(0, np.sqrt(p))
        ax.set_xlabel("pairwise distance")
        ax.set_title("distance distribution")
        ax.legend(loc="upper left", frameon=False)
        fig.tight_layout()
        paths.append(_save(fig, out_dir / "distance_distribution.png"))

    # ------------------------------------------------------------------
    # Validation experiments (Branin): points + RMSE boxplot
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch4_validation.R",
            inputs={"noop.csv": pd.DataFrame({"x": [0.0]})},
            args=[],
            required_packages=["MaxPro", "SFDesign", "rkriging"],
        )
        D = out["design_D.csv"].to_numpy()
        A = out["design_A.csv"].to_numpy()
        rmse = out["rmse.csv"]["rmse"].to_numpy()
        rmse0 = float(out["rmse0.csv"]["rmse0"].iloc[0])

        N_plot = 250
        p1 = np.linspace(0, 1, N_plot)
        p2 = np.linspace(0, 1, N_plot)
        fc = np.zeros((N_plot, N_plot), float)
        for i in range(N_plot):
            for j in range(N_plot):
                fc[i, j] = _branin(np.array([p1[i], p2[j]]))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes[0]
        im = ax.imshow(
            fc.T,
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap=plt.cm.cool_r,
            aspect="auto",
        )
        ax.scatter(D[:, 0], D[:, 1], s=60, color="blue")
        ax.scatter(A[:, 0], A[:, 1], s=80, facecolors="none", edgecolors="darkred", marker="^", linewidths=2)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_title("Validation points")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1]
        ax.boxplot(rmse, widths=0.5)
        ax.axhline(rmse0, color="darkred", linewidth=3)
        ax.set_title("RMSE")
        ax.text(1.05, rmse0, "sequential MaxPro", color="darkred")
        fig.tight_layout()
        paths.append(_save(fig, out_dir / "validation_experiment.png"))

    # ------------------------------------------------------------------
    # Constrained designs (candidate generation in Python, MaxProAugment via R)
    # ------------------------------------------------------------------
    x1c, x2c = _constraint_contour()
    N = int(1e5)
    lhs_all = _lhs_random(N, 2, seed=1)
    gval = np.apply_along_axis(_constraint_vals, 1, lhs_all)
    out_idx = (gval > 0).any(axis=1)
    lhs_feasible = lhs_all[~out_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.scatter(lhs_all[:, 0], lhs_all[:, 1], s=2, color="green", marker="^", alpha=0.6)
    for i in range(3):
        ax.plot(x1c[i, :], x2c[i, :], color="black", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("initial points")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    ax = axes[1]
    ax.scatter(lhs_feasible[:, 0], lhs_feasible[:, 1], s=3, color="red", alpha=0.8)
    for i in range(3):
        ax.plot(x1c[i, :], x2c[i, :], color="black", linewidth=1)
    ax.set_xlim(0.3, 0.8)
    ax.set_ylim(0.35, 0.55)
    ax.set_title("candidate points")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    ax = axes[2]
    if use_r and lhs_feasible.shape[0] > 1:
        exist = lhs_feasible[[0], :]
        cand = lhs_feasible[1:, :]
        out = run_wrapper(
            "ch4_maxpro_augment.R",
            inputs={
                "exist.csv": pd.DataFrame(exist, columns=["x1", "x2"]),
                "cand.csv": pd.DataFrame(cand, columns=["x1", "x2"]),
            },
            args=[str(19)],
            required_packages=["MaxPro"],
        )
        Dmax = out["design.csv"].to_numpy()
        ax.scatter(Dmax[:, 0], Dmax[:, 1], s=70, color="blue")
    for i in range(3):
        ax.plot(x1c[i, :], x2c[i, :], color="black", linewidth=1)
    ax.set_xlim(0.3, 0.8)
    ax.set_ylim(0.35, 0.55)
    ax.set_title("MaxPro design")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    fig.tight_layout()
    paths.append(_save(fig, out_dir / "constrained_designs.png"))

    # ------------------------------------------------------------------
    # Qualitative factors (Random LHD vs SLHD vs MaxProQQ)
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch4_qualitative.R",
            inputs={"noop.csv": pd.DataFrame({"x": [0.0]})},
            args=[],
            required_packages=["SLHD", "MaxPro", "SFDesign"],
        )
        D0 = out["random_lhd.csv"].to_numpy()
        D1q = out["slhd.csv"].to_numpy()
        D2q = out["maxproqq.csv"].to_numpy()

        def _plot_qual(ax: plt.Axes, D: np.ndarray, title: str) -> None:
            if title != "MaxPro":
                q = D[:, 0].astype(int)
                x = D[:, 1:3]
            else:
                x = D[:, 0:2]
                q = D[:, 2].astype(int)

            markers = {1: "o", 2: "^", 3: "s"}
            colors = {1: "red", 2: "green", 3: "blue"}
            for k in [1, 2, 3]:
                idx = q == k
                ax.scatter(x[idx, 0], x[idx, 1], s=180, marker=markers[k], color=colors[k], linewidths=2)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(title, fontsize=14)
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")

            n = 9
            for i in range(1, n + 1):
                t = i / n
                ax.plot([t, t], [0, 1], color="black", linewidth=1, linestyle="--", alpha=0.6)
                ax.plot([0, 1], [t, t], color="black", linewidth=1, linestyle="--", alpha=0.6)
            for i in range(0, n // 3 + 1):
                t = 3 * i / n
                ax.plot([t, t], [0, 1], color="black", linewidth=2)
                ax.plot([0, 1], [t, t], color="black", linewidth=2)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        _plot_qual(axes[0], D0, "random LHD")
        _plot_qual(axes[1], D1q, "SLHD")
        _plot_qual(axes[2], D2q, "MaxPro")
        fig.tight_layout()
        paths.append(_save(fig, out_dir / "qualitative_factors.png"))

    # ------------------------------------------------------------------
    # Nested designs (MaxPro-based)
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch4_nested.R",
            inputs={"noop.csv": pd.DataFrame({"x": [0.0]})},
            args=[],
            required_packages=["MaxPro", "SFDesign"],
        )
        Dfull = out["nested_full.csv"].to_numpy()
        Dsub = out["nested_subset.csv"].to_numpy()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(Dfull[:, 0], Dfull[:, 1], s=30, color="blue")
        ax.scatter(Dsub[:, 0], Dsub[:, 1], s=90, facecolors="none", edgecolors="red", marker="^", linewidths=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("MaxPro-based nested design", fontsize=14)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        fig.tight_layout()
        paths.append(_save(fig, out_dir / "nested_design.png"))

    # ------------------------------------------------------------------
    # Branching designs (R: MaxProQQ + optimization; Python: scatter matrix)
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch4_branching.R",
            inputs={"noop.csv": pd.DataFrame({"x": [0.0]})},
            args=[],
            required_packages=["MaxPro", "SFDesign"],
        )
        D = out["branching_design.csv"].to_numpy()
        fig = _scatter_matrix(
            D,
            var_labels=["branch", "nested1", "nested2", "shared1", "shared2", "shared3", "shared4", "shared5", "shared6"],
            bins=10,
        )
        paths.append(_save(fig, out_dir / "branching_design.png"))

    # ------------------------------------------------------------------
    # Bayesian computation (banana): MED via mined + banana density grid
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch4_bayesian_mined.R",
            inputs={"noop.csv": pd.DataFrame({"x": [0.0]})},
            args=[],
            required_packages=["mined", "MaxPro", "SFDesign"],
        )
        grid = out["banana_grid.csv"]
        ini = out["banana_ini.csv"].to_numpy()
        med = out["banana_med.csv"].to_numpy()
        cand = out["banana_cand.csv"].to_numpy()

        # 2-panel plot: initial MaxPro design vs MED points with annealing
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, pts, title, show_cand in [
            (axes[0], ini, "initial MaxPro design", False),
            (axes[1], med, "MED points with annealing", True),
        ]:
            # density image from grid
            # grid is long-form from R expand.grid(p1, p2) which varies p1 first
            # reshape to (len(p2), len(p1)) for correct imshow orientation
            p1 = np.sort(grid["p1"].unique())
            p2 = np.sort(grid["p2"].unique())
            fc = grid["fc"].to_numpy().reshape(len(p2), len(p1), order="C")
            ax.imshow(fc, origin="lower", extent=(0, 1, 0, 1), cmap=plt.cm.Blues, aspect="auto")
            if show_cand:
                # show candidate points excluding initial ones (match ch4.R intent)
                ax.scatter(cand[:, 0], cand[:, 1], s=10, color="blue", alpha=0.6)
            ax.scatter(ini[:, 0], ini[:, 1], s=50, color="darkred")
            if show_cand:
                ax.scatter(med[:, 0], med[:, 1], s=50, color="darkred")
            ax.set_xlabel(r"$\theta_1$")
            ax.set_ylabel(r"$\theta_2$")
            ax.set_title(title)
        fig.tight_layout()
        paths.append(_save(fig, out_dir / "bayesian_mined.png"))

    # ------------------------------------------------------------------
    # Bayesian computation (banana): exact marginals + MCMC samples
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch4_bayesian_mcmc.R",
            inputs={"noop.csv": pd.DataFrame({"x": [0.0]})},
            args=[],
            required_packages=["cubature", "adaptMCMC"],
        )
        exact1 = out["exact_theta1.csv"]
        exact2 = out["exact_theta2.csv"]
        theta = out["mcmc_samples.csv"].to_numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # density from samples
        import scipy.stats as _st

        for k, ax, exact, xlabel in [
            (0, axes[0], exact1, r"$\theta_1$"),
            (1, axes[1], exact2, r"$\theta_2$"),
        ]:
            xs = exact["p"].to_numpy()
            ax.plot(xs, exact["exact"].to_numpy(), color="green", linewidth=3, label="True")
            kde = _st.gaussian_kde(theta[:, k])
            ax.plot(xs, kde(xs), color="red", linewidth=2, label="MCMC")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("density")
            ax.legend(loc="upper left", frameon=False)
        fig.tight_layout()
        paths.append(_save(fig, out_dir / "bayesian_mcmc_marginals.png"))

    # ------------------------------------------------------------------
    # Candidate generation (CoMinED): full vs feasible candidate set
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch4_comined.R",
            inputs={"noop.csv": pd.DataFrame({"x": [0.0]})},
            args=[],
            required_packages=[],
        )
        cand = out["comined_cand.csv"].to_numpy()
        feas = out["comined_feasible.csv"].to_numpy()
        contour = out["constraint_contour.csv"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, pts, title, xlim, ylim, col in [
            (axes[0], cand, "full candidate set", (0, 1), (0, 1), "blue"),
            (axes[1], feas, "feasible candidate set", (0.3, 0.8), (0.35, 0.55), "green"),
        ]:
            ax.scatter(pts[:, 0], pts[:, 1], s=8, color=col, alpha=0.8)
            for r in [1, 2, 3]:
                c = contour[contour["row"] == r]
                ax.plot(c["x1"].to_numpy(), c["x2"].to_numpy(), color="red", linewidth=2)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_title(title)
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
        fig.tight_layout()
        paths.append(_save(fig, out_dir / "comined_candidates.png"))

    # ------------------------------------------------------------------
    # Minimum energy designs (zip code example in the book uses zipcodeR; excluded here)
    # We still implement the MinED machinery on a synthetic candidate set to keep the pipeline complete.
    # ------------------------------------------------------------------
    if use_r:
        rng = np.random.default_rng(1)
        N = 2000
        CAND = rng.random((N, 2))
        # synthetic positive weights (stand-in for population counts)
        w = 1.0 + 10.0 * (0.3 + 0.7 * (CAND[:, 0] * (1 - CAND[:, 1])))
        out_mined = run_wrapper(
            "ch4_mined_select.R",
            inputs={
                "candidates.csv": pd.DataFrame(CAND, columns=["x1", "x2"]),
                "weights.csv": pd.DataFrame({"w": w}),
            },
            args=[str(200)],
            required_packages=["mined"],
        )
        mined_pts = out_mined["points.csv"].to_numpy()

        out_mm = run_wrapper(
            "ch4_maximin_cand.R",
            inputs={
                "candidates.csv": pd.DataFrame(CAND, columns=["x1", "x2"]),
                "initial.csv": pd.DataFrame(CAND[[0], :], columns=["x1", "x2"]),
            },
            args=[str(199)],
            required_packages=["maximin"],
        )
        maximin_pts = out_mm["points.csv"].to_numpy()

        # Plot side-by-side (note about zipcodeR exclusion)
        sizecode = 0.25 + np.log(w + 1.0) / np.max(np.log(w + 1.0))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, title, pts in [
            (axes[0], "maximin design (synthetic; zipcodeR excluded)", maximin_pts),
            (axes[1], "minimum energy design (synthetic; zipcodeR excluded)", mined_pts),
        ]:
            ax.scatter(CAND[:, 0], CAND[:, 1], s=30 * sizecode, color="pink", alpha=0.8)
            ax.scatter(pts[:, 0], pts[:, 1], s=30, color="blue")
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("")
            ax.set_ylabel("")
        fig.tight_layout()
        paths.append(_save(fig, out_dir / "minimum_energy_designs.png"))

    # ------------------------------------------------------------------
    # Continuous fidelity parameter (R for book-faithful OKfit + rkriging baseline; Python plotting)
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch4_continuous_fidelity.R",
            inputs={"noop.csv": pd.DataFrame({"x": [0.0]})},
            args=[],
            required_packages=["MaxPro", "SFDesign", "rkriging"],
        )
        D = out["design.csv"].to_numpy()
        true = out["true.csv"]["true"].to_numpy()
        pred_mf = out["pred_multifidelity.csv"]["pred"].to_numpy()
        pred_sf = out["pred_singlefidelity.csv"]["pred"].to_numpy()
        rmse_mf = float(out["rmse.csv"]["rmse_multifidelity"].iloc[0])
        rmse_sf = float(out["rmse.csv"]["rmse_singlefidelity"].iloc[0])

        fig = _scatter_matrix(D, var_labels=[r"$x_1$", r"$x_2$", "t"], bins=15)
        paths.append(_save(fig, out_dir / "continuous_fidelity_scatter_matrix.png"))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, pred, title, rmse in [
            (axes[0], pred_mf, "DoIt (continuous fidelity)", rmse_mf),
            (axes[1], pred_sf, "Single-fidelity simulation", rmse_sf),
        ]:
            ax.scatter(true, pred, s=12, color="blue", alpha=0.7)
            lo = min(true.min(), pred.min())
            hi = max(true.max(), pred.max())
            ax.plot([lo, hi], [lo, hi], color="black", linewidth=2)
            ax.set_xlabel("true")
            ax.set_ylabel("pred")
            ax.set_title(f"{title}\\nRMSE={rmse:.4g}")
        fig.tight_layout()
        paths.append(_save(fig, out_dir / "continuous_fidelity_predictions.png"))

    return out_dir, paths


def main(argv: Sequence[str] | None = None) -> int:
    generate_figures(use_r=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


