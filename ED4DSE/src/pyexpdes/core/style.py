from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl


@dataclass(frozen=True)
class RStyle:
    """
    Small helper to approximate base R plotting defaults used in the book scripts.

    We keep this intentionally minimal and explicit per-figure (font sizes, line widths, etc.)
    rather than forcing a global style that might surprise users.
    """

    # Rough mapping of R's "cex" usage in the scripts
    cex_axis: float = 2.0
    cex_lab: float = 2.0
    cex_main: float = 2.0

    line_width: float = 3.0


def apply_base_style() -> None:
    """
    Apply a conservative matplotlib style to keep figures deterministic and clean.
    """

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "font.family": "DejaVu Sans",
        }
    )


