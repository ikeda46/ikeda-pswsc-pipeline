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


def plot_summary(spec: dict, df_pwv: "pd.DataFrame", tstar: dict, title: str | None = None,
                  out_path: str | None = None, figsize: tuple[float, float] = (11, 12),
                  flagged_freq_raw: "np.ndarray | None" = None) -> "plt.Figure":
    """Combined 3-panel figure (one PNG): (1) ON-OFF diff spectrum, (2)
    OFF-point PWV(t), (3) eta_atm-corrected T* spectrum -- all as plain
    lines, deliberately WITHOUT error bars/shading (unlike `plot_spectrum`)
    for a cleaner look when the three are shown together.

    spec: from `spectrum.onoff_diff_spectrum`.
    df_pwv: from `pwv.estimate_pwv_timeseries`.
    tstar: from `pwv.tstar_spectrum` (also carries its own `flagged`,
    the low-mean-eta_atm channels excluded from the T* panel).
    flagged_freq_raw: same as `plot_spectrum`'s `flagged_freq` -- raw-noisy
    KIDs excluded before the spectrum was built, marked on panel 1 only.
    """
    fig, (ax_spec, ax_pwv, ax_tstar) = plt.subplots(3, 1, figsize=figsize)

    order = np.argsort(spec["freq"])
    ax_spec.plot(spec["freq"][order], spec["mean"][order], "o", ms=2, color="tab:blue")
    ax_spec.axhline(0, color="k", lw=0.5)
    if flagged_freq_raw is not None and len(flagged_freq_raw):
        ax_spec.plot(flagged_freq_raw, np.zeros(len(flagged_freq_raw)), "x", ms=5, mew=1.2, color="red",
                     label="flagged (noisy KID, excluded)")
        ax_spec.legend(fontsize=8)
    ax_spec.set_xlabel("Frequency [GHz]")
    ax_spec.set_ylabel("tb_on - tb_off [K]")
    ax_spec.set_title("ON-OFF diff spectrum (LPF 1.0Hz)")

    df_pwv_sorted = df_pwv.sort_values("t")
    t0 = df_pwv_sorted["t"].min()
    ax_pwv.plot(df_pwv_sorted["t"] - t0, df_pwv_sorted["pwv"], "-o", ms=2, lw=0.8, color="tab:green")
    ax_pwv.set_xlabel("Time since observation start [s]")
    ax_pwv.set_ylabel("PWV [mm]")
    ax_pwv.set_title("OFF-point PWV (model eta_atm, 60s window / 20s step)")

    order_t = np.argsort(tstar["freq"])
    ax_tstar.plot(tstar["freq"][order_t], tstar["tstar"][order_t], "o", ms=2, color="tab:purple")
    ax_tstar.axhline(0, color="k", lw=0.5)
    flagged_freq_t = tstar["freq"][tstar["flagged"]]
    if len(flagged_freq_t):
        eta_min = np.nanmax(tstar["eta_mean"][tstar["flagged"]]) if tstar["flagged"].any() else None
        label = f"flagged (mean eta_atm < ~{eta_min:.2f})" if eta_min is not None else "flagged"
        ax_tstar.plot(flagged_freq_t, np.zeros(len(flagged_freq_t)), "x", ms=5, mew=1.2, color="red", label=label)
        ax_tstar.legend(fontsize=8)
    ax_tstar.set_xlabel("Frequency [GHz]")
    ax_tstar.set_ylabel("T* [K]")
    ax_tstar.set_title("T* = (ON-OFF diff spectrum) / mean eta_atm")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
