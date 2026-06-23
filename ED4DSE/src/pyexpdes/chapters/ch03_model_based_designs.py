"""
Python port of `R files/ch3.R` (pure Python).

The R script covers:
- Random 2D designs (3 panels)
- 1D RMSE curves for equi-spaced, Chebyshev, IMSE-optimal, MMSE-optimal
- IMSE/MMSE and relative efficiency curves vs theta
- Maximum entropy designs (determinant criterion)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize

from pyexpdes.core.paths import figures_dir
from pyexpdes.core.style import apply_base_style


def _kernel_rbf(d: np.ndarray, theta: float) -> np.ndarray:
    return np.exp(-((d / theta) ** 2))


def _rmse_curve(D: np.ndarray, theta: float, test: np.ndarray) -> np.ndarray:
    D = np.asarray(D, float)
    n = D.size
    E = np.abs(D[:, None] - D[None, :])
    R = _kernel_rbf(E, theta)
    L = cholesky(R + 1e-6 * np.eye(n), lower=False)

    def s(x: float) -> float:
        r = np.exp(-(((x - D) / theta) ** 2))
        a = solve_triangular(L.T, r, lower=True)
        v = 1.0 - float(np.sum(a * a))
        return float(np.sqrt(max(v, 0.0)))

    return np.array([s(float(t)) for t in test])


def _imse(D: np.ndarray, theta: float) -> float:
    D = np.asarray(D, float)
    n = D.size
    E = np.abs(D[:, None] - D[None, :])
    R = _kernel_rbf(E, theta)
    L = cholesky(R + 1e-6 * np.eye(n), lower=False)

    def s2(x: float) -> float:
        r = np.exp(-(((x - D) / theta) ** 2))
        a = solve_triangular(L.T, r, lower=True)
        v = 1.0 - float(np.sum(a * a))
        return float(max(v, 0.0))

    # R uses integrate(Vectorize(s), 0,1, abs.tol=1e-10) on s2()
    # Use simple dense trapezoid for determinism.
    grid = np.linspace(0.0, 1.0, 2001)
    vals = np.array([s2(float(g)) for g in grid])
    return float(trapezoid(vals, grid))


def _mmse(D: np.ndarray, theta: float) -> float:
    test = np.linspace(0.0, 1.0, 1001)
    sa = _rmse_curve(D, theta, test)
    return float(np.max(sa * sa))


def _optimize_design(
    D0: np.ndarray, theta: float, objective: str
) -> np.ndarray:
    D0 = np.asarray(D0, float)
    bounds = [(0.0, 1.0)] * D0.size

    if objective == "imse":
        f = lambda x: _imse(x, theta)
    elif objective == "mmse":
        f = lambda x: _mmse(x, theta)
    elif objective == "negent":
        f = lambda x: -_entropy(x, theta)
    else:
        raise ValueError(objective)

    res = minimize(f, D0, method="L-BFGS-B", bounds=bounds)
    return np.asarray(res.x, float)


def _entropy(D: np.ndarray, theta: float) -> float:
    D = np.asarray(D, float)
    E = np.abs(D[:, None] - D[None, :])
    R = _kernel_rbf(E, theta)
    return float(np.linalg.det(R))


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_figures(*, use_r: bool = True) -> tuple[Path, list[Path]]:
    apply_base_style()
    out_dir = figures_dir("ch03")
    paths: list[Path] = []

    # ------------------------------------------------------------------
    # Random designs (p=2, n=7, 1x3, seeds 9,45,31)
    # ------------------------------------------------------------------
    p = 2
    n = 7
    seeds = [9, 45, 31]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, seed in zip(axes, seeds):
        rng = np.random.default_rng(seed)
        D = rng.random((n, p))
        ax.scatter(D[:, 0], D[:, 1], s=200, color="blue")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$x_1$", fontsize=14)
        ax.set_ylabel(r"$x_2$", fontsize=14)
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "random_designs.png"))

    # ------------------------------------------------------------------
    # RMSE curves (theta=0.1, n=10, 2x2)
    # ------------------------------------------------------------------
    n = 10
    theta = 0.1
    test = np.linspace(0.0, 1.0, 301)

    D1 = (np.arange(1, n + 1) - 1) / (n - 1)  # equi-spaced
    d = np.cos((2 * (np.arange(1, n + 1)) - 1) / (2 * n) * np.pi)
    D2 = (d + 1) / 2  # Chebyshev
    D0 = (np.arange(1, n + 1) - 0.5) / n
    D3 = _optimize_design(D0, theta, "imse")
    D4 = _optimize_design(D0, theta, "mmse")

    designs = [
        ("equi-spaced design", D1),
        ("Chebyshev design", D2),
        ("IMSE-optimal design", D3),
        ("MMSE-optimal design", D4),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    for ax, (title, D) in zip(axes, designs):
        rmse = _rmse_curve(D, theta, test)
        ax.plot(test, rmse, linewidth=3)
        ax.scatter(D, np.zeros_like(D), s=80, color="blue")
        ax.set_ylim(0, 0.7)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("RMSE", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.text(0.2, 0.67, f"IMSE={_imse(D, theta):.4f}", color="red")
        ax.text(0.75, 0.67, f"MMSE={_mmse(D, theta):.4f}", color="green")
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "rmse_panels.png"))

    # ------------------------------------------------------------------
    # IMSE/MMSE/RE vs theta (2x2)
    # ------------------------------------------------------------------
    ev = np.linspace(0.01, 0.4, 100)
    IR = np.zeros((ev.size, 4))
    MR = np.zeros((ev.size, 4))
    for i, th in enumerate(ev):
        IR[i, :] = [_imse(D1, th), _imse(D2, th), _imse(D3, th), _imse(D4, th)]
        MR[i, :] = [_mmse(D1, th), _mmse(D2, th), _mmse(D3, th), _mmse(D4, th)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    labels = ["equi-spaced", "Chebyshev", "IMSE-optimal", "MMSE-optimal"]

    axes[0, 0].plot(ev, IR, linewidth=2)
    axes[0, 0].set_title("IMSE")
    axes[0, 0].set_xlabel(r"$\theta$")
    axes[0, 0].set_ylabel("IMSE")
    axes[0, 0].legend(labels, frameon=False, fontsize=9)

    axes[0, 1].plot(ev, MR, linewidth=2)
    axes[0, 1].set_title("MMSE")
    axes[0, 1].set_xlabel(r"$\theta$")
    axes[0, 1].set_ylabel("MMSE")

    IRR = IR / IR.min(axis=1, keepdims=True)
    MRR = MR / MR.min(axis=1, keepdims=True)
    axes[1, 0].plot(ev, 1.0 / IRR, linewidth=2)
    axes[1, 0].set_title("Relative efficiency-IMSE")
    axes[1, 0].set_xlabel(r"$\theta$")
    axes[1, 0].set_ylabel("RE")

    axes[1, 1].plot(ev, 1.0 / MRR, linewidth=2)
    axes[1, 1].set_title("Relative efficiency-MMSE")
    axes[1, 1].set_xlabel(r"$\theta$")
    axes[1, 1].set_ylabel("RE")

    fig.tight_layout()
    paths.append(_save(fig, out_dir / "relative_efficiency.png"))

    # ------------------------------------------------------------------
    # Maximum entropy designs (1x3, theta=0.1, n=10)
    # ------------------------------------------------------------------
    theta = 0.1
    test = np.linspace(0.0, 1.0, 301)
    D3 = _optimize_design(D0, theta, "imse")
    D4 = _optimize_design(D0, theta, "mmse")
    D5 = _optimize_design((np.arange(1, n + 1) - 1) / (n - 1), theta, "negent")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (title, D) in zip(
        axes,
        [
            ("IMSE-optimal design", D3),
            ("MMSE-optimal design", D4),
            ("Maximum entropy design", D5),
        ],
    ):
        rmse = _rmse_curve(D, theta, test)
        ax.plot(test, rmse, linewidth=3)
        ax.scatter(D, np.zeros_like(D), s=100, color="blue")
        ax.set_ylim(0, 0.7)
        ax.set_title(title, fontsize=14)
        ax.text(0.5, 0.67, f"Entropy={_entropy(D, theta):.4f}", color="purple", ha="center")
        ax.text(0.5, 0.62, f"IMSE={_imse(D, theta):.4f}", color="red", ha="center")
        ax.text(0.5, 0.57, f"MMSE={_mmse(D, theta):.4f}", color="green", ha="center")
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "entropy_designs.png"))

    return out_dir, paths


def main(argv: Sequence[str] | None = None) -> int:
    generate_figures()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


