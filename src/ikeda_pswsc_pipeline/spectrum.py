"""Step 1: build spectra from simultaneous ON-OFF differencing.

Before Step 0, beam B's calibration was believed too unstable to trust, so
spectrum-building was centered on beam A alone -- which means the
SIMULTANEOUS off-source beam could not be used, and methods had to fall
back on differencing beam A's own on-state and off-state samples (which
occur ~60s apart, not at the same time) or on summary-statistic
differences.

Step 0 (`calibration.step0_calibrate_beamB`) calibrates beam B directly
against beam A's Tb from GRAD, with the regularization strength chosen by
cross-validation -- this is trustworthy enough to use beam B as the
SIMULTANEOUS off-source reference for beam A's on-source signal (and vice
versa), which is what this module demonstrates at several LPF settings.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from d24_tools import utils as d24_utils

from .calibration import apply_lpf_per_beam, load_pswsc
from .faica_denoise import fit_faica

ON_BEAM_FOR_STATE = {"ON": "A", "OFF": "B"}
OFF_BEAM_FOR_STATE = {"ON": "B", "OFF": "A"}

MIN_SAMPLES_PER_WINDOW = 4
WIN_SEC = 60.0


def onoff_diff_spectrum_from_tb(tb: np.ndarray, state: np.ndarray, beam: np.ndarray, t_sec: np.ndarray,
                                 chan_common: np.ndarray, freq_common: np.ndarray, obsid: str, obj_name: str,
                                 win_sec: float = WIN_SEC,
                                 min_samples_per_window: int = MIN_SAMPLES_PER_WINDOW) -> dict:
    """Core windowed simultaneous ON-OFF differencing, given an
    already-built (n_samples, n_chan_common) calibrated Tb array (`tb`,
    aligned to `chan_common`/`freq_common`) plus the file's own
    state/beam/t_sec arrays. Split out of `onoff_diff_spectrum` so other
    Tb-construction methods (e.g. FA-reconstructed, see
    `fa_ica_spectrum`) can reuse the same windowing/differencing logic.

    For each ON/OFF block, at each `win_sec` window: the off-beam's Tb is
    spline-interpolated onto the on-beam's own sample times (both beams'
    samples occur within the same window, interleaved but not
    simultaneous), then subtracted directly -- no atmospheric model
    anywhere. Per-window means are aggregated (mean +/- std) into the
    final per-channel spectrum.
    """
    change = np.where(state[1:] != state[:-1])[0] + 1
    bounds = np.concatenate(([0], change, [len(state)]))
    blocks = [(state[bounds[i]], bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    blocks = [(s, lo, hi) for s, lo, hi in blocks if s in ("ON", "OFF")]

    rows = []
    for s, lo, hi in blocks:
        on_beam = ON_BEAM_FOR_STATE[s]
        off_beam = OFF_BEAM_FOR_STATE[s]
        on_mask = beam == on_beam
        off_mask = beam == off_beam
        block_idx_full = np.arange(lo, hi)
        t0_b, t1_b = t_sec[lo], t_sec[hi - 1]
        n_windows = max(1, round((t1_b - t0_b) / win_sec))
        edges = np.linspace(t0_b, t1_b + 1e-6, n_windows + 1)
        for w in range(n_windows):
            in_win = (t_sec[block_idx_full] >= edges[w]) & (t_sec[block_idx_full] < edges[w + 1])
            idx_on = block_idx_full[in_win & on_mask[block_idx_full]]
            idx_off = block_idx_full[in_win & off_mask[block_idx_full]]
            if idx_on.size < min_samples_per_window or idx_off.size < min_samples_per_window:
                continue
            t_off_w = t_sec[idx_off]
            t_on_w = t_sec[idx_on]
            tb_off_w = tb[idx_off]
            tb_on_w = tb[idx_on]
            keep = (t_on_w >= t_off_w[0]) & (t_on_w <= t_off_w[-1])
            if keep.sum() < min_samples_per_window:
                continue
            cs = CubicSpline(t_off_w, tb_off_w, axis=0)
            resid = tb_on_w[keep] - cs(t_on_w[keep])
            rows.append(np.nanmean(resid, axis=0))

    arr = np.array(rows)
    return dict(obsid=obsid, obj_name=obj_name, freq=freq_common, chan=chan_common,
                mean=np.nanmean(arr, axis=0), std=np.nanstd(arr, axis=0), n=arr.shape[0])


def onoff_diff_spectrum(target_path: str, a_A: np.ndarray, b_A: np.ndarray,
                         a_B: np.ndarray, b_B: np.ndarray, chan_common: np.ndarray,
                         lpf_cutoff_hz: float | None, win_sec: float = WIN_SEC,
                         min_samples_per_window: int = MIN_SAMPLES_PER_WINDOW) -> dict:
    """Simultaneous ON-OFF-differenced spectrum.

    a_A/b_A/a_B/b_B: per-channel calibration for beam A / beam B (beam B's
    normally from `step0_calibrate_beamB`), matched to `chan_common`.

    lpf_cutoff_hz: None = no filtering at all (raw per-sample Tb); a float
    = per-beam LPF at that cutoff (see `apply_lpf_per_beam`) applied to
    each beam's own raw timestream BEFORE calibration.

    See `onoff_diff_spectrum_from_tb` for the windowing/differencing logic.
    """
    da_sub = load_pswsc(target_path)
    obsid = str(da_sub.aste_obs_id)
    obj_name = str(da_sub.aste_obs_file).split("_")[0]

    state = da_sub.state.to_numpy()
    beam = da_sub.beam.to_numpy()
    chan_all = da_sub.chan.to_numpy()
    freq_all = da_sub.frequency.to_numpy()
    t_sec = d24_utils.dt_to_seconds(da_sub)
    x_raw = da_sub.to_numpy()

    x_use = apply_lpf_per_beam(t_sec, beam, x_raw, cutoff_hz=lpf_cutoff_hz) if lpf_cutoff_hz else x_raw

    _, idx_obs, _ = np.intersect1d(chan_all, chan_common, return_indices=True)
    freq_common = freq_all[idx_obs]

    tb = np.full((x_use.shape[0], len(chan_common)), np.nan)
    selA = beam == "A"
    selB = beam == "B"
    tb[selA] = a_A[None, :] * x_use[np.ix_(selA, idx_obs)] + b_A[None, :]
    tb[selB] = a_B[None, :] * x_use[np.ix_(selB, idx_obs)] + b_B[None, :]

    return onoff_diff_spectrum_from_tb(tb, state, beam, t_sec, chan_common, freq_common,
                                        obsid, obj_name, win_sec, min_samples_per_window)


def fa_ica_spectrum(target_path: str, a_A: np.ndarray, b_A: np.ndarray,
                     a_B: np.ndarray, b_B: np.ndarray, chan_common: np.ndarray,
                     n_components: int = 20, win_sec: float = WIN_SEC,
                     min_samples_per_window: int = MIN_SAMPLES_PER_WINDOW) -> dict:
    """Third spectrum-construction method: same windowed ON-OFF
    differencing as `onoff_diff_spectrum`, but built from the
    FA(m=20)+JADE-RECONSTRUCTED Tb time series instead of the raw or
    LPF'd one.

    Fits FA+JADE separately per beam (on that beam's own 1.0Hz-LPF'd,
    ON+OFF-only calibrated Tb, `faica_denoise.fit_faica`) and reconstructs
    using ALL m components (`A_mix @ S + mean`) -- i.e. the FA low-rank
    signal model with each channel's own idiosyncratic noise
    (`fa.noise_variance_`) dropped and everything explained by the m
    common factors kept. No per-IC "strange IC" removal is applied (that
    approach was found to have a side effect of increasing noise
    elsewhere -- see `faica_denoise.flag_noisy_kids`'s docstring); this is
    a plain full reconstruction, shown purely as one of three methods to
    compare, not as an adopted denoising step.
    """
    da_sub = load_pswsc(target_path)
    obsid = str(da_sub.aste_obs_id)
    obj_name = str(da_sub.aste_obs_file).split("_")[0]

    state = da_sub.state.to_numpy()
    beam = da_sub.beam.to_numpy()
    chan_all = da_sub.chan.to_numpy()
    freq_all = da_sub.frequency.to_numpy()
    t_sec = d24_utils.dt_to_seconds(da_sub)
    x_raw = da_sub.to_numpy()

    x_lpf = apply_lpf_per_beam(t_sec, beam, x_raw, cutoff_hz=1.0)
    _, idx_obs, _ = np.intersect1d(chan_all, chan_common, return_indices=True)
    freq_common = freq_all[idx_obs]

    onoff_mask = (state == "ON") | (state == "OFF")
    tb_fa = np.full((x_lpf.shape[0], len(chan_common)), np.nan)
    for bm, (a_bm, b_bm) in [("A", (a_A, b_A)), ("B", (a_B, b_B))]:
        sel = (beam == bm) & onoff_mask
        tb_bm = a_bm[None, :] * x_lpf[np.ix_(sel, idx_obs)] + b_bm[None, :]
        fit = fit_faica(tb_bm, n_components=n_components)
        tb_fa[sel] = (fit["A_mix"] @ fit["S"]).T + fit["mean"][None, :]

    return onoff_diff_spectrum_from_tb(tb_fa, state, beam, t_sec, chan_common, freq_common,
                                        obsid, obj_name, win_sec, min_samples_per_window)
