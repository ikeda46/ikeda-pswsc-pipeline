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
                   figsize: tuple[float, float] = (11, 4), ax: "plt.Axes | None" = None,
                   flagged_freq: "np.ndarray | None" = None) -> "plt.Figure":
    """Plot one spectrum dict as its own independent figure (mean +/- std
    vs frequency, ON-OFF=0 reference line).

    ax: if given, draw into this existing Axes instead of creating a new
    Figure (still returns the Axes' figure) -- lets a caller build a
    multi-figure grid of otherwise-independent plots if it wants to, while
    every OTHER caller just gets one clean standalone figure per call.

    flagged_freq: frequencies of KIDs excluded before this spectrum was
    built (e.g. `chan_flagged_raw` from `step0_calibrate_beamB`) -- since
    they have no data point in `spec`, they're marked as small red x's
    sitting on the Tb=0 line instead.
    """
    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    order = np.argsort(spec["freq"])
    ax.errorbar(spec["freq"][order], spec["mean"][order], yerr=spec["std"][order],
                fmt="o", ms=3, lw=0.8, capsize=2, color="tab:blue", ecolor="tab:blue", alpha=0.8)
    ax.axhline(0, color="k", lw=0.5)
    if flagged_freq is not None and len(flagged_freq):
        ax.plot(flagged_freq, np.zeros(len(flagged_freq)), "x", ms=5, mew=1.2, color="red",
                 label="flagged (noisy KID, excluded)")
        ax.legend(fontsize=8)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("tb_on - tb_off [K]")
    if title:
        ax.set_title(title)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        if owns_fig:
            plt.close(fig)
    return fig
