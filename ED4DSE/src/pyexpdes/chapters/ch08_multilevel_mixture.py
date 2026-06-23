"""
Python port of `R files/Ch8.R` (Fractional Factorial Designs).

Sections covered:
- 8.1 Two-level Designs (half-normal plot)
- 8.2 Bayesian-inspired Designs (Kriging + HiGarrote)
- 8.3 Designs with more than two levels (L18, MaxPro, Energy)
- 8.4 Mixture designs (Simplex, Ternary plots)

Python-first policy:
- Plotting is done in Python (matplotlib).
- R is invoked for: rkriging, HiGarrote, DoE.base, twinning, mixexp, support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from pyexpdes.core.paths import figures_dir, datasets_dir
from pyexpdes.core.style import apply_base_style
from pyexpdes.core.r_utils import run_wrapper


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _ternary_to_cartesian(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert ternary coordinates (p1, p2, p3) to Cartesian (x, y).
    p1 + p2 + p3 = 1 (simplex constraint).
    """
    x = 0.5 * (2 * p2 + p3)
    y = (np.sqrt(3) / 2) * p3
    return x, y


def _draw_ternary_axes(ax: plt.Axes) -> None:
    """Draw ternary plot axes and grid."""
    # Triangle vertices
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(vertices[:, 0], vertices[:, 1], "k-", linewidth=1.5)

    # Grid lines
    for i in range(1, 10):
        frac = i / 10
        # Lines parallel to each edge
        x1, y1 = _ternary_to_cartesian(1 - frac, frac, 0)
        x2, y2 = _ternary_to_cartesian(0, frac, 1 - frac)
        ax.plot([x1, x2], [y1, y2], "gray", linewidth=0.3, alpha=0.5)

        x1, y1 = _ternary_to_cartesian(1 - frac, 0, frac)
        x2, y2 = _ternary_to_cartesian(0, 1 - frac, frac)
        ax.plot([x1, x2], [y1, y2], "gray", linewidth=0.3, alpha=0.5)

        x1, y1 = _ternary_to_cartesian(frac, 1 - frac, 0)
        x2, y2 = _ternary_to_cartesian(frac, 0, 1 - frac)
        ax.plot([x1, x2], [y1, y2], "gray", linewidth=0.3, alpha=0.5)

    # Labels
    ax.text(0, -0.08, r"$x_1$", fontsize=14, ha="center")
    ax.text(1, -0.08, r"$x_2$", fontsize=14, ha="center")
    ax.text(0.5, np.sqrt(3) / 2 + 0.05, r"$x_3$", fontsize=14, ha="center")

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.15, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")


def generate_figures(*, use_r: bool = True) -> tuple[Path, list[Path]]:
    """
    Generate Chapter 8 figures.

    Returns (output_dir, [figure_paths...]).
    """
    apply_base_style()
    out_dir = figures_dir("ch08")
    paths: list[Path] = []

    # ------------------------------------------------------------------
    # Figure 8.1: Half-normal plot (Two-level design) - Pure Python
    # ------------------------------------------------------------------
    # 2^(4-1) fractional factorial design
    A = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    B = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    C = np.array([-1, 1, -1, 1, -1, 1, -1, 1])
    D = A * B * C  # D = ABC (defining relation)

    # Response data
    y = np.array([504, 984, 928, 808, 992, 784, 464, 976])

    # Build design matrix for main effects and 2-way interactions
    X = np.column_stack([
        np.ones(8),
        A, B, C,
        A * B, A * C, B * C,
        A * B * C
    ])

    # Fit linear model (OLS)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    eff = beta[1:]  # Exclude intercept
    eff_names = ["1", "2", "3", "12", "13", "23", "123"]

    # Half-normal plot
    I = len(eff)
    u = norm.ppf(0.5 + 0.5 * (np.arange(1, I + 1) - 0.5) / I)
    sorted_idx = np.argsort(np.abs(eff))
    sorted_abs_eff = np.abs(eff)[sorted_idx]
    sorted_names = [eff_names[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (ui, effi, name) in enumerate(zip(u, sorted_abs_eff, sorted_names)):
        color = "red" if i >= I - 3 else "blue"
        ax.text(ui, effi, name, fontsize=14, ha="center", va="bottom", color=color)

    ax.set_xlim(0, max(u) + 0.2)
    ax.set_ylim(0, max(sorted_abs_eff) * 1.1)
    ax.set_xlabel("half-normal quantiles", fontsize=14)
    ax.set_ylabel("absolute coefficients", fontsize=14)
    ax.set_title("Half-normal plot", fontsize=16)

    fig.tight_layout()
    paths.append(_save(fig, out_dir / "halfnormal_twolevel.png"))

    # ------------------------------------------------------------------
    # Figure 8.2: Bayesian-inspired half-normal plot
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch8_halfnormal_bayesian.R",
            inputs={
                "design.csv": pd.DataFrame({"A": A, "B": B, "C": C, "D": D}),
                "response.csv": pd.DataFrame({"y": y}),
            },
            args=[],
            required_packages=["rkriging", "HiGarrote"],
        )
        bayesian_eff = out["bayesian_effects.csv"]

        eff_vals = bayesian_eff["effect"].values
        eff_names_b = bayesian_eff["name"].values

        I = len(eff_vals)
        u = norm.ppf(0.5 + 0.5 * (np.arange(1, I + 1) - 0.5) / I)
        sorted_idx = np.argsort(np.abs(eff_vals))
        sorted_abs_eff = np.abs(eff_vals)[sorted_idx]
        sorted_names = [eff_names_b[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, (ui, effi, name) in enumerate(zip(u, sorted_abs_eff, sorted_names)):
            color = "red" if i >= I - 3 else "blue"
            size = 14 if i >= I - 3 else 12
            ax.text(ui, effi, name, fontsize=size, ha="center", va="bottom", color=color)

        ax.set_xlim(0, max(u) + 0.2)
        ax.set_ylim(0, max(sorted_abs_eff) * 1.1)
        ax.set_xlabel("half-normal quantiles", fontsize=14)
        ax.set_ylabel("absolute coefficients", fontsize=14)
        ax.set_title("Half-normal plot (Bayesian)", fontsize=16)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "halfnormal_bayesian.png"))

    # ------------------------------------------------------------------
    # Cast Fatigue Experiment Analysis (HiGarrote)
    # ------------------------------------------------------------------
    if use_r:
        cast_path = datasets_dir() / "cast.txt"
        if cast_path.exists():
            # Read cast data
            cast = pd.read_csv(cast_path, sep=r"\s+")

            out = run_wrapper(
                "ch8_cast_fatigue.R",
                inputs={
                    "cast.csv": cast,
                },
                args=[],
                required_packages=["HiGarrote"],
            )

            hg_result = out["higarrote_result.csv"]
            full_model = out["full_model.csv"]
            best_model = out["best_model.csv"]
            model_comp = out["model_comparison.csv"]

            # Create summary figure
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Left: HiGarrote selected variables
            ax = axes[0]
            vars_selected = hg_result[hg_result["coefficient"].abs() > 0.01]
            bars = ax.barh(vars_selected["var_name"], vars_selected["coefficient"], color="steelblue")
            ax.set_xlabel("Coefficient", fontsize=14)
            ax.set_title("HiGarrote Variable Selection (Cast Fatigue)", fontsize=14)
            ax.axvline(x=0, color="black", linewidth=0.5)

            # Right: Model comparison
            ax = axes[1]
            x_labels = ["Full Model", "Best Model (HiGarrote)"]
            r2_vals = model_comp["r_squared"].values
            adj_r2_vals = model_comp["adj_r_squared"].values

            x = np.arange(len(x_labels))
            width = 0.35
            bars1 = ax.bar(x - width/2, r2_vals, width, label=r"$R^2$", color="steelblue")
            bars2 = ax.bar(x + width/2, adj_r2_vals, width, label=r"Adj $R^2$", color="coral")

            ax.set_ylabel("Value", fontsize=14)
            ax.set_title("Model Comparison", fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels)
            ax.legend(fontsize=12)
            ax.set_ylim(0, 1.0)

            fig.tight_layout()
            paths.append(_save(fig, out_dir / "cast_fatigue_analysis.png"))

    # ------------------------------------------------------------------
    # Figure 8.3: MaxPro and Energy distance for 18-run designs
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch8_orthogonal_array.R",
            inputs={"params.csv": pd.DataFrame({"n_sim": [100000], "seed": [1]})},
            args=[],
            required_packages=["DoE.base", "SFDesign", "twinning"],
        )
        sim_results = out["simulation_results.csv"]
        design_metrics = out["design_metrics.csv"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # MaxPro plot
        ax = axes[0]
        mp_sim = sim_results["maxpro"].values
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(mp_sim)
        x_vals = np.linspace(mp_sim.min(), mp_sim.max(), 200)
        ax.plot(x_vals, kde(x_vals), "k-", linewidth=2)
        ax.fill_between(x_vals, kde(x_vals), alpha=0.3, color="gray")

        # Vertical lines for L18 designs
        for _, row in design_metrics.iterrows():
            ax.axvline(row["maxpro"], color="blue", linewidth=2, alpha=0.7)

        # Special designs
        d1_mp = design_metrics[design_metrics["design"] == "D1"]["maxpro"].values[0]
        d2_mp = design_metrics[design_metrics["design"] == "D2"]["maxpro"].values[0]
        ax.axvline(d1_mp, color="red", linewidth=3, linestyle=":")
        ax.axvline(d2_mp, color="green", linewidth=3, linestyle="--")
        ax.text(d1_mp, kde([d1_mp])[0] * 0.8, r"$D_1$", fontsize=14)
        ax.text(d2_mp, kde([d2_mp])[0] * 0.8, r"$D_2$", fontsize=14)

        ax.set_xlabel("MaxPro measure", fontsize=14)
        ax.set_ylabel("density", fontsize=14)
        ax.set_title("MaxPro criterion for 18-run designs", fontsize=14)

        # Energy plot
        ax = axes[1]
        en_sim = sim_results["energy"].values
        kde = gaussian_kde(en_sim)
        x_vals = np.linspace(en_sim.min(), en_sim.max(), 200)
        ax.plot(x_vals, kde(x_vals), "k-", linewidth=2)
        ax.fill_between(x_vals, kde(x_vals), alpha=0.3, color="gray")

        for _, row in design_metrics.iterrows():
            ax.axvline(row["energy"], color="blue", linewidth=2, alpha=0.7)

        d1_en = design_metrics[design_metrics["design"] == "D1"]["energy"].values[0]
        d2_en = design_metrics[design_metrics["design"] == "D2"]["energy"].values[0]
        ax.axvline(d1_en, color="red", linewidth=3, linestyle=":")
        ax.axvline(d2_en, color="green", linewidth=3, linestyle="--")
        ax.text(d1_en, kde([d1_en])[0] * 0.8, r"$D_1$", fontsize=14)
        ax.text(d2_en, kde([d2_en])[0] * 0.8, r"$D_2$", fontsize=14)

        ax.set_xlabel("energy distance", fontsize=14)
        ax.set_ylabel("density", fontsize=14)
        ax.set_title("Energy distance for 18-run designs", fontsize=14)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "maxpro_energy_comparison.png"))

    # ------------------------------------------------------------------
    # Figure 8.4: Mixture designs (SLD and SCD) - Ternary plots
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch8_mixture_design.R",
            inputs={"params.csv": pd.DataFrame({"n_sld": [3], "degree": [2], "seed": [1]})},
            args=[],
            required_packages=["mixexp", "spacefillr", "support"],
        )
        sld_design = out["sld_design.csv"].to_numpy()
        scd_design = out["scd_design.csv"].to_numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # SLD plot
        ax = axes[0]
        _draw_ternary_axes(ax)
        x, y_plot = _ternary_to_cartesian(sld_design[:, 0], sld_design[:, 1], sld_design[:, 2])
        ax.scatter(x, y_plot, s=150, color="blue", zorder=5)
        ax.set_title("Simplex Lattice Design (SLD)", fontsize=14)

        # SCD plot
        ax = axes[1]
        _draw_ternary_axes(ax)
        x, y_plot = _ternary_to_cartesian(scd_design[:, 0], scd_design[:, 1], scd_design[:, 2])
        ax.scatter(x, y_plot, s=150, color="blue", zorder=5)
        ax.set_title("Simplex Centroid Design (SCD)", fontsize=14)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "mixture_sld_scd.png"))

    # ------------------------------------------------------------------
    # Figure 8.5: Space-filling mixture designs - Ternary plots
    # ------------------------------------------------------------------
    if use_r:
        out = run_wrapper(
            "ch8_mixture_design.R",
            inputs={"params.csv": pd.DataFrame({"n_sld": [3], "degree": [2], "seed": [1]})},
            args=[],
            required_packages=["mixexp", "spacefillr", "support"],
        )
        sf_design = out["spacefilling_design.csv"].to_numpy()
        sf_constrained = out["spacefilling_constrained.csv"].to_numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Unconstrained space-filling
        ax = axes[0]
        _draw_ternary_axes(ax)
        x, y_plot = _ternary_to_cartesian(sf_design[:, 0], sf_design[:, 1], sf_design[:, 2])
        ax.scatter(x, y_plot, s=150, color="blue", zorder=5)
        ax.set_title("Space-filling design (unconstrained)", fontsize=14)

        # Constrained space-filling
        ax = axes[1]
        _draw_ternary_axes(ax)
        x, y_plot = _ternary_to_cartesian(sf_constrained[:, 0], sf_constrained[:, 1], sf_constrained[:, 2])
        ax.scatter(x, y_plot, s=150, color="blue", zorder=5)
        ax.set_title("Space-filling design (constrained)", fontsize=14)

        fig.tight_layout()
        paths.append(_save(fig, out_dir / "mixture_spacefilling.png"))

    return out_dir, paths


def main(argv: Sequence[str] | None = None) -> int:
    generate_figures(use_r=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
