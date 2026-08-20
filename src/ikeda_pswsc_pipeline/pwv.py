"""Step 2/3: OFF-point PWV(t) estimation, and eta_atm-corrected T* spectrum.

Unlike the model-free ON-OFF diff spectrum (`spectrum.py`), estimating PWV
requires the physical ATM model (`d24_tools.atm.get_eta_atm`, TOPTICA
filter-convolved via `d24_tools.parallel.atm_avg_filter`) -- the model-free
method has no PWV concept at all. This reuses the exact same PWV-fit
machinery as the production skydip pipeline
(`tau0/skydip_channel_spectra_v3.py`'s `_filter_setup`/`_fit_chunk_pwv`,
same pattern `grad_skydip.py` already uses) rather than re-deriving it.

Per-window PWV fit source: BOTH beams' off-point samples are pooled within
a window, whichever beam is off-source at that instant (`state=="ON" &
beam=="B"` or `state=="OFF" & beam=="A"`) -- not restricted to one beam.
This mirrors `faica_denoise`'s "blank" data definition elsewhere in this
pipeline, and is valid here specifically because Step 0 calibrates beam B
to already AGREE with beam A's Tb (that agreement is Step 0's whole
purpose), so mixing samples from both beams within one window is no
longer the beam-offset-contaminated mess it would have been pre-Step-0.

eta_atm(chan) for the T* correction is the PWV(t)-averaged (not
instantaneous) per-channel value -- deliberately, since a per-window
divide-by-eta_atm would blow up at any single low-eta_atm window/pwv
combination; averaging first makes near-zero values far less likely, and
any that remain are flagged (`ETA_MIN`) rather than plotted.

freq>250GHz split (2026-08-19): `_filter_setup`'s FREQ_MIN_GHZ=250 cut
(inherited from the original daisy `empirical_tau0.py` PWV fit) exists
to keep the PWV curve_fit itself well-conditioned -- channels below
250GHz sit where the atmosphere is more transparent, so eta_atm has
little PWV-leverage there, and including them dilutes/destabilizes the
single-parameter fit. It is NOT a data-availability limit: the TOPTICA
filter measurement (`d24_tools.utils.load_filters`) covers 200-459.9GHz,
well below 250GHz. So the restriction only needs to apply to the PWV
FIT (`estimate_pwv_timeseries`, unchanged, still >250GHz-only for fit
stability) -- once PWV(t) is known, eta_atm at any OTHER frequency can
still be evaluated from the same model, just not used to help determine
that PWV. `mean_eta_atm_per_chan` therefore evaluates eta_atm (and so
T*) for the FULL common channel set by default, decoupled from the
fit's own channel restriction -- see `_filter_setup_full` below.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from d24_tools import atm, utils as d24_utils

sys.path.insert(0, "../../tau0")
from skydip_channel_spectra_v3 import _filter_setup, _fit_chunk_pwv  # noqa: E402

from .calibration import apply_lpf_per_beam, load_pswsc

WIN_SEC = 60.0
STEP_SEC = 20.0
MIN_SAMPLES_PER_WINDOW = 10
ETA_MIN = 0.2


def estimate_pwv_timeseries(target_path: str, a_A: np.ndarray, b_A: np.ndarray,
                             a_B: np.ndarray, b_B: np.ndarray, chan_common: np.ndarray,
                             win_sec: float = WIN_SEC, step_sec: float = STEP_SEC,
                             min_samples: int = MIN_SAMPLES_PER_WINDOW) -> pd.DataFrame:
    """Sliding-window (width=win_sec, step=step_sec, overlapping) PWV fit
    from the OFF-point beam's calibrated Tb (model eta_atm, freq>250GHz
    channels per `_filter_setup`). Returns a DataFrame (t, pwv, el,
    t_amb, n), one row per window that converged, sorted by t.
    """
    da_sub = load_pswsc(target_path)
    state = da_sub.state.to_numpy()
    beam = da_sub.beam.to_numpy()
    chan_all = da_sub.chan.to_numpy()
    freq_all = da_sub.frequency.to_numpy()
    t_amb = da_sub.temperature.to_numpy().astype(np.float64)
    t_sec = d24_utils.dt_to_seconds(da_sub)
    el = da_sub.lat.to_numpy() / 3600.0
    x_raw = da_sub.to_numpy()

    x_lpf = apply_lpf_per_beam(t_sec, beam, x_raw, cutoff_hz=1.0)
    _, idx_obs, _ = np.intersect1d(chan_all, chan_common, return_indices=True)
    freq_common = freq_all[idx_obs]

    tb = np.full((x_lpf.shape[0], len(chan_common)), np.nan)
    selA = beam == "A"
    selB = beam == "B"
    tb[selA] = a_A[None, :] * x_lpf[np.ix_(selA, idx_obs)] + b_A[None, :]
    tb[selB] = a_B[None, :] * x_lpf[np.ix_(selB, idx_obs)] + b_B[None, :]

    is_off = ((state == "ON") & selB) | ((state == "OFF") & selA)

    gamma, eta_fdf, f_mid, freq_fit, idx_fit = _filter_setup(chan_common, freq_common)
    atm_interp = atm.get_eta_atm()

    t0, t1 = t_sec[is_off].min(), t_sec[is_off].max()
    starts = np.arange(t0, t1, step_sec)

    rows = []
    for t_start in starts:
        in_win = is_off & (t_sec >= t_start) & (t_sec < t_start + win_sec)
        idx = np.where(in_win)[0]
        if idx.size < min_samples:
            continue
        spec = np.nanmean(tb[idx][:, idx_fit], axis=0)
        ok = np.isfinite(spec)
        if ok.sum() < min_samples:
            continue
        tamb_k = float(np.nanmean(t_amb[idx]))
        el_k = float(np.nanmean(el[idx]))
        csc_k = 1.0 / np.sin(np.deg2rad(el_k))
        t_mid = float(np.nanmean(t_sec[idx]))
        pwv = _fit_chunk_pwv(spec[ok], tamb_k, csc_k, freq_fit[ok], f_mid, eta_fdf[:, ok], gamma[ok], atm_interp)
        if pwv is None:
            continue
        rows.append(dict(t=t_mid, pwv=pwv, el=el_k, t_amb=tamb_k, n=int(idx.size)))

    return pd.DataFrame(rows).sort_values("t").reset_index(drop=True)


def _filter_setup_full(chan_k: np.ndarray, freq_k: np.ndarray):
    """Same TOPTICA-filter-convolution setup as `_filter_setup`, but
    WITHOUT its freq>250GHz mask -- returns (gamma, eta_fdf, f_mid,
    freq, idx_obs) for every channel in `chan_k` that has TOPTICA filter
    data (i.e. the full ~200-459.9GHz range), not just the >250GHz
    subset `_filter_setup` keeps for PWV-fit stability. Used to evaluate
    eta_atm at an ALREADY-KNOWN pwv across the full band (see module
    docstring) -- never for fitting pwv itself, which still goes through
    `_filter_setup`/`_fit_chunk_pwv` unchanged.
    """
    eta_f, chan_toptica, _, freq_toptica = d24_utils.load_filters()
    eta_f = eta_f ** 2
    _, idx_top, idx_obs = np.intersect1d(chan_toptica, chan_k, assume_unique=True, return_indices=True)
    eta_f_sel = eta_f[:, idx_top]
    freq_sel = freq_k[idx_obs]

    df = np.diff(freq_toptica)[:, None]
    eta_fdf = (eta_f_sel[:-1] + eta_f_sel[1:]) / 2 * df
    gamma = np.nansum(eta_fdf, axis=0)
    f_mid = (freq_toptica[1:] + freq_toptica[:-1]) / 2
    return gamma, eta_fdf, f_mid, freq_sel, idx_obs


def mean_eta_atm_per_chan(df_pwv: pd.DataFrame, chan_common: np.ndarray, freq_common: np.ndarray,
                           eta_min: float = ETA_MIN, full_range: bool = True) -> dict:
    """Per-channel eta_atm, averaged over the whole PWV(t) time series
    (`estimate_pwv_timeseries`'s output) -- a single value per channel,
    not a time series. Channels whose mean eta_atm < eta_min are flagged
    (deep absorption-line troughs -- dividing by these would blow up).

    full_range=True (default): evaluates eta_atm for every channel in
    `chan_common` that has TOPTICA filter data (`_filter_setup_full`),
    i.e. matches the full spectrum's own frequency range, NOT just the
    freq>250GHz subset used to fit the pwv this is evaluated at (see
    module docstring for why that's a valid thing to do).
    full_range=False: restricted to the same freq>250GHz subset as the
    PWV fit (`_filter_setup`) -- kept only for comparison/backward
    compatibility, not the default going forward.
    """
    filter_setup = _filter_setup_full if full_range else _filter_setup
    gamma, eta_fdf, f_mid, freq_fit, idx_fit = filter_setup(chan_common, freq_common)
    atm_interp = atm.get_eta_atm()

    csc = 1.0 / np.sin(np.deg2rad(df_pwv["el"].to_numpy()))
    eta_per_window = np.array([
        np.nansum(eta_fdf * (atm_interp(f_mid, pwv).squeeze() ** c)[:, None], axis=0) / gamma
        for pwv, c in zip(df_pwv["pwv"].to_numpy(), csc)
    ])
    eta_mean = np.nanmean(eta_per_window, axis=0)
    flagged = eta_mean < eta_min
    return dict(chan=chan_common[idx_fit], freq=freq_fit, idx_fit=idx_fit,
                eta_mean=eta_mean, flagged=flagged, eta_min=eta_min)


def tstar_spectrum(spec: dict, eta_result: dict) -> dict:
    """T* = (ON-OFF diff spectrum) / (mean eta_atm), per channel --
    `spec` from `spectrum.onoff_diff_spectrum` (its `chan`/`freq` must be
    the SAME `chan_common` passed to `mean_eta_atm_per_chan`, same order,
    which is always true when both come from the same Step 0 `chan`).
    Flagged (low mean eta_atm) channels are NaN, not divided.
    """
    idx_fit = eta_result["idx_fit"]
    flagged = eta_result["flagged"]
    tstar = np.full(idx_fit.shape, np.nan)
    ok = ~flagged
    tstar[ok] = spec["mean"][idx_fit][ok] / eta_result["eta_mean"][ok]
    return dict(freq=eta_result["freq"], chan=eta_result["chan"], tstar=tstar,
                flagged=flagged, eta_mean=eta_result["eta_mean"])
