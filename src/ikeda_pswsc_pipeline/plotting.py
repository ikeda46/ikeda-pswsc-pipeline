"""Display: turn a spectrum dict (from `spectrum.onoff_diff_spectrum`/
`fa_ica_spectrum`, or anything with the same `freq`/`mean`/`std` shape)
into its own independent figure.

Kept separate from spectrum construction so a pipeline reads as
calibration -> spectrum construction -> display, with display being a
single reusable function instead of ad hoc subplot code repeated per
script.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_spectrum(spec: dict, title: str | None = None, out_path: str | None = None,
                   figsize: tuple[float, float] = (11, 4), ax: "plt.Axes | None" = None) -> "plt.Figure":
    """Plot one spectrum dict as its own independent figure (mean +/- std
    vs frequency, ON-OFF=0 reference line).

    ax: if given, draw into this existing Axes instead of creating a new
    Figure (still returns the Axes' figure) -- lets a caller build a
    multi-figure grid of otherwise-independent plots if it wants to, while
    every OTHER caller just gets one clean standalone figure per call.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    order = np.argsort(spec["freq"])
    ax.errorbar(spec["freq"][order], spec["mean"][order], yerr=spec["std"][order],
                fmt="o", ms=3, lw=0.8, capsize=2, color="tab:blue", ecolor="tab:blue", alpha=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("tb_on - tb_off [K]")
    if title:
        ax.set_title(title)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig
