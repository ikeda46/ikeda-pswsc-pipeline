"""In-situ, per-beam skydip-style calibration from GRAD off-point data.

2026-08-10 finding (see `calibration.py`'s module docstring): GRAD is NOT
purely blank sky. Because beam A and beam B are separated by a fixed
azimuth "beam throw" much larger than the beam FWHM, whichever beam a
GRAD chunk transitions AWAY FROM (having just been on-source in the
preceding ON/OFF block) starts that chunk near-source and only becomes
genuinely blank ("off-point") in the chunk's later portion, as the antenna
finishes slewing to the *other* beam's on-source position.

This module identifies, per beam, the 10 (of pswsc's usual 21) GRAD
chunks where that beam ends up at its own off-point right before the next
block starts, keeps only the LATTER half (by time) of each such chunk
(the settled, genuinely-blank portion), and fits a per-beam calibration
from that pooled off-point data using the same PWV-fit-then-per-channel-
regression method as the production skydip pipeline
(`tau0/skydip_channel_spectra_v3.py`) -- except the elevation-bin grouping
there is replaced by these 10 physical chunks directly (no artificial
elevation binning needed: with only ~3.9deg of elevation drift over a
pswsc file, an el-bin grid would just be an arbitrary re-slicing of the
same 10 chunks).

Method, per beam:
1. Bootstrap: seed PWV from ALMA (`d24_tools.atm.get_pwv`), fit an initial
   GLOBAL per-channel (a, b) by regressing raw signal against this
   seed-PWV's model Tb, pooled over all off-point samples.
2. Per chunk: average the round-1-calibrated Tb (freq>250GHz channels
   only, matching the production pipeline's PWV-fit channel selection)
   and fit that CHUNK's own PWV against the ATM model
   (`skydip_channel_spectra_v3._fit_chunk_pwv`) -- letting PWV vary
   chunk-to-chunk captures real atmospheric drift over the ~20min
   observation, rather than assuming one PWV for the whole file.
3. Final: using each sample's OWN chunk's fitted PWV to compute that
   sample's model Tb target, pool ALL off-point samples (not
   chunk-averaged -- thousands of samples, not just 10 points) and do one
   per-channel OLS regression (raw vs Tb target) for the final (a, b).

This is a single bootstrap pass (unlike the production pipeline's 2-round
iteration) -- adequate for this comparison; add a second round if the
chunk-PWV values look unstable in practice.
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.stats import linregress

from d24_tools import atm, utils as d24_utils

sys.path.insert(0, "../../tau0")
from skydip_channel_spectra_v3 import _filter_setup, _fit_chunk_pwv  # noqa: E402

from .calibration import load_pswsc, flag_noisy_kids_raw, RAW_NOISY_KID_THRESH

OFFPOINT_TIME_FRAC = 0.5  # use the LAST 50% (by time) of each qualifying GRAD chunk


def _offpoint_grad_chunk_indices(state: np.ndarray, t_sec: np.ndarray, beam: np.ndarray,
                                  beam_of_interest: str,
                                  offpoint_time_frac: float = OFFPOINT_TIME_FRAC) -> list[np.ndarray]:
    """Sample indices (restricted to `beam_of_interest`) from the latter
    `offpoint_time_frac` portion of each GRAD chunk that transitions INTO
    the state where `beam_of_interest` is off-source (OFF for beam A,
    ON for beam B, since ON_BEAM_FOR_STATE={"ON":"A","OFF":"B"}).

    Returns a list of index arrays, one per qualifying chunk (there are
    normally 10 of pswsc's 21 GRAD chunks for a given beam; the very first
    GRAD chunk, the observation's startup transient, has no predecessor
    block and is never included in either beam's list).
    """
    dest_state = "OFF" if beam_of_interest == "A" else "ON"
    change = np.where(state[1:] != state[:-1])[0] + 1
    bounds = np.concatenate(([0], change, [len(state)]))
    blocks = [(state[bounds[i]], bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    idx_list = []
    for i, (s, lo, hi) in enumerate(blocks):
        if s != "GRAD":
            continue
        nxt = blocks[i + 1][0] if i + 1 < len(blocks) else None
        if nxt != dest_state:
            continue
        idx = np.arange(lo, hi)
        idx = idx[beam[idx] == beam_of_interest]
        if idx.size == 0:
            continue
        t = t_sec[idx]
        cutoff = t[0] + (t[-1] - t[0]) * (1 - offpoint_time_frac)
        idx = idx[t >= cutoff]
        if idx.size > 0:
            idx_list.append(idx)
    return idx_list


def _per_channel_ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x, y: (n_samples, n_chan). Returns (a, b, R2) per channel."""
    n_chan = x.shape[1]
    a = np.full(n_chan, np.nan)
    b = np.full(n_chan, np.nan)
    r2 = np.full(n_chan, np.nan)
    for c in range(n_chan):
        xc, yc = x[:, c], y[:, c]
        ok = np.isfinite(xc) & np.isfinite(yc)
        if ok.sum() < 10:
            continue
        res = linregress(xc[ok], yc[ok])
        a[c], b[c], r2[c] = res.slope, res.intercept, res.rvalue ** 2
    return a, b, r2


def pwv_seed_from_calibrated_tb(target_path: str, beam_of_interest: str, a_ext: np.ndarray, b_ext: np.ndarray,
                                 chan_ext: np.ndarray, offpoint_time_frac: float = OFFPOINT_TIME_FRAC) -> float:
    """PWV seed fit from an ALREADY-calibrated Tb (e.g. beam A's existing
    external skydip (a, b)) applied to this file's own off-point GRAD data
    for `beam_of_interest` -- an alternative to the ALMA-seeded PWV used by
    default in `calibrate_beam_from_grad_offpoint`, to check whether the
    bootstrap seed itself (rather than dilution/attenuation bias) explains
    the observed slope mismatch against the external calibration.
    """
    da_sub = load_pswsc(target_path)
    state = da_sub.state.to_numpy()
    beam = da_sub.beam.to_numpy()
    chan_all = da_sub.chan.to_numpy()
    freq_all = da_sub.frequency.to_numpy()
    t_sec = d24_utils.dt_to_seconds(da_sub)
    x_raw = da_sub.to_numpy()
    el_all = da_sub.lat.to_numpy() / 3600.0
    tamb_all = da_sub.temperature.to_numpy().astype(np.float64)

    chunk_idx_list = _offpoint_grad_chunk_indices(state, t_sec, beam, beam_of_interest, offpoint_time_frac)
    idx_all = np.concatenate(chunk_idx_list)

    _, idx_obs_ext, idx_cal_ext = np.intersect1d(chan_all, chan_ext, return_indices=True)
    chan_common = chan_all[idx_obs_ext]
    freq_common = freq_all[idx_obs_ext]
    a_c = a_ext[idx_cal_ext]
    b_c = b_ext[idx_cal_ext]

    tb = a_c[None, :] * x_raw[np.ix_(idx_all, idx_obs_ext)] + b_c[None, :]

    gamma2, eta_fdf2, f_mid2, freq_fit, idx_obs2_masked = _filter_setup(chan_common, freq_common)
    spec = np.nanmean(tb[:, idx_obs2_masked], axis=0)
    tamb_ck = float(np.nanmean(tamb_all[idx_all]))
    el_pts = el_all[idx_all]
    csc_ck = float(np.nanmean(1.0 / np.sin(el_pts * np.pi / 180.0)))

    atm_interp = atm.get_eta_atm()
    pwv = _fit_chunk_pwv(spec, tamb_ck, csc_ck, freq_fit, f_mid2, eta_fdf2, gamma2, atm_interp)
    if pwv is None:
        raise ValueError("PWV fit against externally-calibrated Tb failed to converge")
    return pwv


def calibrate_beam_from_grad_offpoint(target_path: str, beam_of_interest: str,
                                       offpoint_time_frac: float = OFFPOINT_TIME_FRAC,
                                       raw_noise_thresh: float = RAW_NOISY_KID_THRESH,
                                       pwv_seed_override: float | None = None) -> dict:
    """In-situ skydip-style calibration of one beam ('A' or 'B') from its
    own off-point GRAD data (see module docstring).

    Raw-noisy KIDs are excluded first (`calibration.flag_noisy_kids_raw`:
    beam-A-only, >1Hz raw-residual STD exceeding `raw_noise_thresh`x the
    median), same as Step 0 -- a pathologically noisy channel would
    otherwise corrupt this fit just as it would Step 0's CV.

    pwv_seed_override: if given, used as the bootstrap round-1 PWV instead
    of the ALMA-seeded value (e.g. a PWV fit from beam A's existing
    external skydip calibration applied to this same off-point data --
    see `pwv_seed_from_calibrated_tb`).
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
    el_all = da_sub.lat.to_numpy() / 3600.0
    tamb_all = da_sub.temperature.to_numpy().astype(np.float64)

    flagged_raw = flag_noisy_kids_raw(t_sec, beam, x_raw, thresh_factor=raw_noise_thresh)

    chunk_idx_list = _offpoint_grad_chunk_indices(state, t_sec, beam, beam_of_interest, offpoint_time_frac)
    if len(chunk_idx_list) == 0:
        raise ValueError(f"no off-point GRAD chunks found for beam {beam_of_interest}")
    idx_all = np.concatenate(chunk_idx_list)
    chunk_id_of_sample = np.concatenate([np.full(len(idx), i) for i, idx in enumerate(chunk_idx_list)])
    n_chunks = len(chunk_idx_list)

    # channel selection: intersect with the TOPTICA filter-shape channels
    # (same selection the production skydip pipeline uses), excluding
    # raw-noisy KIDs
    eta_f, chan_toptica, _, freq_toptica = d24_utils.load_filters()
    eta_f = eta_f ** 2
    _, idx_top, idx_obs = np.intersect1d(chan_toptica, chan_all, assume_unique=True, return_indices=True)
    keep = ~flagged_raw[idx_obs]
    idx_top, idx_obs = idx_top[keep], idx_obs[keep]
    eta_f_sel = eta_f[:, idx_top]
    chan_sel = chan_all[idx_obs]
    freq_sel = freq_all[idx_obs]

    gamma = np.nansum((eta_f_sel[:-1] + eta_f_sel[1:]) / 2 * np.diff(freq_toptica)[:, None], axis=0)
    eta_fdf = (eta_f_sel[:-1] + eta_f_sel[1:]) / 2 * np.diff(freq_toptica)[:, None]
    f_toptica_mid = (freq_toptica[1:] + freq_toptica[:-1]) / 2

    x_pts = x_raw[np.ix_(idx_all, idx_obs)]
    el_pts = el_all[idx_all]
    csc_pts = 1.0 / np.sin(el_pts * np.pi / 180.0)
    tamb_pts = tamb_all[idx_all]

    atm_interp = atm.get_eta_atm()

    # --- bootstrap: seed PWV -> round-1 (a, b) -----------------------------
    if pwv_seed_override is not None:
        pwv_seed = float(pwv_seed_override)
    else:
        pwv_seed = float(np.nanmean(atm.get_pwv(t_sec[idx_all])))
    eta_atm_seed = atm_interp(f_toptica_mid, pwv_seed).squeeze()[None, :] ** csc_pts[:, None]
    eta_atm_los_seed = (eta_atm_seed @ eta_fdf) / gamma[None, :]
    tb_target_round1 = (1 - eta_atm_los_seed) * tamb_pts[:, None]
    a1, b1, r2_1 = _per_channel_ols(x_pts, tb_target_round1)

    tb_round1 = a1[None, :] * x_pts + b1[None, :]

    # --- per-chunk PWV fit from round-1 calibrated Tb ---------------------
    gamma2, eta_fdf2, f_mid2, freq_fit, idx_obs2_masked = _filter_setup(chan_sel, freq_sel)

    pwv_chunks = np.full(n_chunks, np.nan)
    tamb_chunks = np.full(n_chunks, np.nan)
    for i in range(n_chunks):
        sel = chunk_id_of_sample == i
        spec_ck = np.nanmean(tb_round1[sel][:, idx_obs2_masked], axis=0)
        tamb_ck = float(np.nanmean(tamb_pts[sel]))
        csc_ck = float(np.nanmean(csc_pts[sel]))
        tamb_chunks[i] = tamb_ck
        pwv_ck = _fit_chunk_pwv(spec_ck, tamb_ck, csc_ck, freq_fit, f_mid2, eta_fdf2, gamma2, atm_interp)
        pwv_chunks[i] = pwv_ck if pwv_ck is not None else np.nan

    # --- final: per-sample Tb target using ITS OWN chunk's PWV, pooled ---
    ok_chunks = np.isfinite(pwv_chunks)
    keep = ok_chunks[chunk_id_of_sample]
    # per-chunk PWV values aren't sorted/unique, so evaluate one at a time
    # (RectBivariateSpline.__call__ with grid=True requires strictly
    # increasing y) rather than passing the whole array at once.
    eta_atm_chunks = np.array([atm_interp(f_toptica_mid, pwv).squeeze() for pwv in pwv_chunks[ok_chunks]])
    chunk_id_remap = np.full(n_chunks, -1)
    chunk_id_remap[ok_chunks] = np.arange(ok_chunks.sum())

    cid = chunk_id_remap[chunk_id_of_sample[keep]]
    eta_atm_final = eta_atm_chunks[cid] ** csc_pts[keep, None]
    eta_atm_los_final = (eta_atm_final @ eta_fdf) / gamma[None, :]
    tb_target_final = (1 - eta_atm_los_final) * tamb_pts[keep, None]

    a, b, r2 = _per_channel_ols(x_pts[keep], tb_target_final)

    return dict(obsid=obsid, obj_name=obj_name, beam=beam_of_interest, chan=chan_sel, freq=freq_sel,
                a=a, b=b, r2=r2, a_round1=a1, b_round1=b1, r2_round1=r2_1,
                pwv_seed=pwv_seed, pwv_chunks=pwv_chunks, tamb_chunks=tamb_chunks,
                n_chunks=n_chunks, n_samples=int(keep.sum()), n_samples_total=idx_all.size,
                chan_flagged_raw=chan_all[flagged_raw])
