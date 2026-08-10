"""Step 0: calibrate beam B against beam A directly from GRAD data.

pswsc's GRAD state is the brief antenna azimuth-slew transition between
the ON and OFF nod positions (elevation is held to simple target tracking
throughout -- confirmed by comparing the `lat`/`lon` coordinates against
the `lat_origin`/`lon_origin` tracking-reference coordinates). It carries
no elevation excursion, so it cannot support an independent per-beam
calibration fit the way daisy's mini-skydip GRAD segment can
(`d24_tools.utils.get_mini_skydip`/`Tb_mini_skydip`).

What GRAD *can* do: beam A and beam B samples occur within the same short
window, at essentially the same time and pointing, so beam B's own
calibration can be fit directly against beam A's brightness temperature
(computed from beam A's own borrowed skydip calibration). Because this
regression fully determines beam B's calibration, beam B's own
independent skydip calibration is not needed at all -- only beam A's.

Per-channel, this is a simple 2-parameter regression (y=tb_A ~= a*x_B + b).
**Step 0's method** (not a fixed set of numbers) is: ridge-regularize this
regression toward the prior (a_A, b_A) via two dimensionless shrinkage
fractions frac_a/frac_b (see `_ridge_fit`), with frac_a/frac_b chosen by
leave-one-GRAD-block-out CROSS-VALIDATION (`cv_select_frac`) for the
specific file being calibrated -- never fixed/hand-picked, and never
reused as-is from a different observation/session. `step0_calibrate_beamB`
is the single entry point that runs this whole procedure (CV search, then
the final ridge fit at the CV-selected fracs) for one pswsc file.

On the two files tried during development (NGC1068 obsid=20240715094140,
Mars obsid=20240715115449), CV happened to pick frac_a=1.0, frac_b=1e-7
for BOTH -- i.e. the slope (a) needed strong shrinkage toward beam A
(consistent with eta_atm being flat/low-leverage at most frequencies,
making the slope hard to pin down from only ~20 GRAD blocks), while the
intercept (b) was already well-determined and needed essentially no
shrinkage. That is a DATA POINT about this method, not a default to carry
forward -- always re-run the CV search per file.

Before any of this, raw-noisy KIDs are excluded (`flag_noisy_kids_raw`):
the CV grid search picks a SINGLE frac_a/frac_b shared across all
channels by averaging a per-channel score, so one pathologically noisy
channel can degrade the regularization choice for every other (good)
channel -- and this happens before any FA/JADE fit exists to catch it the
way `faica_denoise.flag_noisy_kids` does downstream. This pre-filter
works on the raw (pre-calibration) signal: apply the pipeline's 1.0Hz LPF
and take the residual it REMOVES (`x_raw - x_lpf`, i.e. >1Hz content) as
a receiver-noise proxy uncontaminated by real (sub-1Hz) signal, then flag
a KID if that residual's STD exceeds 10x the median. Computed from beam A
ONLY -- beam B's >1Hz noise floor is systematically ~5x higher across
essentially all channels (a beam-wide receiver-chain effect), which
dilutes a same-beam relative threshold for known-bad channels there, and
beam A is the more trustworthy reference anyway (its calibration is
borrowed directly from skydip, not fit within Step 0). It's a complement
to, not a replacement for, the later FA-based flagging -- it protects the
calibration step; the FA-based one protects the final spectrum estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import linregress

import decode as dc
from d24_tools import reduce, utils

LPF_CUTOFF_HZ = 1.0
LPF_ORDER = 4
MIN_SAMPLES_PER_BEAM = 5
RAW_NOISY_KID_THRESH = 10.0


def load_pswsc(target_path: str) -> "xr.DataArray":
    """Load a pswsc dems file, restrict to GRAD/ON/OFF, despike, sort by time."""
    da = dc.load.dems(target_path)
    if da.long_name != "df/f":
        da = dc.load.dems(target_path, data_scaling="df/f")

    arr = da.to_numpy()
    bad = np.isnan(np.mean(arr, axis=0)) | (np.nanstd(arr, axis=0) == 0)
    da = da[:, ~bad]

    return reduce.despike(da, include=["GRAD", "ON", "OFF"]).sortby("time")


def apply_lpf_per_beam(t_sec: np.ndarray, beam: np.ndarray, x_raw: np.ndarray,
                        cutoff_hz: float = LPF_CUTOFF_HZ, order: int = LPF_ORDER) -> np.ndarray:
    """Low-pass filter each beam's own sample sequence independently.

    Beam A and beam B alternate every ~5-7 raw samples (~13Hz). Filtering
    the mixed raw stream before splitting by beam blends the two beams'
    fast-alternating values together and washes out any genuine
    beam-to-beam contrast -- this must be done per beam, after splitting.
    """
    ok_ch = np.all(np.isfinite(x_raw), axis=0)
    x_lpf = x_raw.copy()
    for bm in ("A", "B"):
        sel = beam == bm
        t_bm = t_sec[sel]
        fs_bm = 1.0 / np.median(np.diff(t_bm))
        b_lpf, a_lpf = butter(order, cutoff_hz, btype="low", fs=fs_bm)
        x_bm = x_raw[sel]
        x_bm_f = x_bm.copy()
        x_bm_f[:, ok_ch] = filtfilt(b_lpf, a_lpf, x_bm[:, ok_ch], axis=0)
        x_lpf[sel] = x_bm_f
    return x_lpf


def _ridge_fit(x: np.ndarray, y: np.ndarray, a0: float, b0: float,
                frac_a: float, frac_b: float) -> tuple[float, float]:
    """Ridge-regularized fit of y ~= a*x + b, penalizing (a-a0)^2 and
    (b-b0)^2 -- pulls the fit toward the prior (a0, b0) when the data
    itself (x, y) has little leverage (e.g. a near-flat/low-SNR channel),
    while leaving the fit close to plain OLS when the data strongly
    constrains the slope/intercept on its own.

    Minimizes sum((y - a*x - b)**2) + lambda_a*(a-a0)**2 + lambda_b*(b-b0)**2,
    with lambda_a = frac_a*Sxx_centered and lambda_b = frac_b*N
    (Sxx_centered = sum((x-mean(x))**2), N = number of points) -- i.e.
    frac_a/frac_b are DIMENSIONLESS shrinkage fractions (0 = plain OLS;
    ~1 = comparable weight to the data itself; large = pinned to the
    prior), automatically scaled to each channel's own data instead of
    needing a raw lambda tuned per channel/target. Sxx_centered (not the
    raw, uncentered sum(x**2)) is used for the scale specifically because
    x (raw df/f) has a large nonzero mean -- the *variance* around that
    mean is what actually constrains the slope, and using the raw
    (mean-inflated) sum(x**2) would make the regularization far too
    aggressive relative to the data's real leverage.
    """
    # x (raw df/f) has a large nonzero mean relative to its actual scatter,
    # so the UNCENTERED normal equations are numerically ill-conditioned
    # (Sxx, Sx both dominated by mean(x)**2/mean(x) terms, nearly singular)
    # -- even a tiny regularization term then has a wildly amplified effect
    # on the solved (a, b). Fit in centered coordinates (xc = x - xbar,
    # c = a*xbar + b) instead, which is well-conditioned, then convert back.
    N = x.size
    k = float(np.mean(x))  # xbar
    xc = x - k
    Sxxc = np.sum(xc * xc)
    Sxcy = np.sum(xc * y)
    Sy = np.sum(y)
    lambda_a = frac_a * Sxxc
    lambda_b = frac_b * N

    denom_a = Sxxc + lambda_a + N * lambda_b * k * k / (N + lambda_b)
    numer_a = Sxcy + lambda_a * a0 + (lambda_b * k / (N + lambda_b)) * (Sy - N * b0)
    a = numer_a / denom_a
    c = (Sy + lambda_b * b0 + lambda_b * a * k) / (N + lambda_b)
    b = c - a * k
    return float(a), float(b)


def _ridge_fit_vec(X: np.ndarray, Y: np.ndarray, a0: np.ndarray, b0: np.ndarray,
                    frac_a: float, frac_b: float) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized (across channels) version of `_ridge_fit`.

    X, Y: (n_blocks, n_channels). a0, b0: (n_channels,). Returns (a, b),
    each (n_channels,). Same centered-coordinate closed form as
    `_ridge_fit`, just with the per-block sums taken along axis=0.
    """
    N = X.shape[0]
    k = np.mean(X, axis=0)  # (n_chan,)
    Xc = X - k[None, :]
    Sxxc = np.sum(Xc * Xc, axis=0)
    Sxcy = np.sum(Xc * Y, axis=0)
    Sy = np.sum(Y, axis=0)
    lambda_a = frac_a * Sxxc
    lambda_b = frac_b * N

    denom_a = Sxxc + lambda_a + N * lambda_b * k * k / (N + lambda_b)
    numer_a = Sxcy + lambda_a * a0 + (lambda_b * k / (N + lambda_b)) * (Sy - N * b0)
    a = numer_a / denom_a
    c = (Sy + lambda_b * b0 + lambda_b * a * k) / (N + lambda_b)
    b = c - a * k
    return a, b


def flag_noisy_kids_raw(t_sec: np.ndarray, beam: np.ndarray, x_raw: np.ndarray,
                         cutoff_hz: float = LPF_CUTOFF_HZ, order: int = LPF_ORDER,
                         thresh_factor: float = RAW_NOISY_KID_THRESH) -> np.ndarray:
    """Flag noisy KIDs from RAW (pre-calibration) data, before Step 0's fit.

    `cv_select_frac` picks a SINGLE frac_a/frac_b shared across all
    channels by averaging a per-channel CV score -- a pathologically noisy
    KID can corrupt that average and degrade the regularization choice for
    every other (good) channel, even though it's the calibration step
    itself, before any FA/JADE fit exists to catch it the way
    `faica_denoise.flag_noisy_kids` does downstream. So this filter must
    run on the raw signal, before calibration.

    Per channel: apply the same 1.0Hz LPF used throughout this pipeline,
    then take the residual REMOVED by the filter (`x_raw - x_lpf`, i.e.
    the >1Hz content) as a receiver-noise proxy. Real
    astrophysical/atmospheric signal varies slower than 1Hz -- that's the
    whole reason 1.0Hz is used as this pipeline's signal-preserving cutoff
    -- so this residual isolates fast noise without being contaminated by
    real signal power the way a plain STD of the raw or LPF'd signal would
    be for bright channels.

    Computed from beam A ONLY (not beam A AND beam B, unlike
    `faica_denoise.flag_noisy_kids`): beam B's >1Hz residual noise is
    systematically ~5x higher than beam A's across essentially ALL
    channels (a beam-wide receiver-chain effect, not a bad-channel
    effect), while the same known-bad channels' excess noise looks like a
    roughly fixed additive amount rather than something that scales with
    the beam's own baseline -- so on beam B, that fixed excess gets
    diluted by the already-high baseline and fails a same-beam relative
    threshold, even though the SAME channels clear it easily on beam A.
    Beam A is also the more trustworthy reference here (its calibration is
    borrowed directly from skydip, not fit within Step 0), so using it
    alone avoids that dilution rather than trying to compensate for it.

    Flags a KID if its beam-A residual STD exceeds `thresh_factor` times
    the median beam-A residual STD.
    """
    x_lpf = apply_lpf_per_beam(t_sec, beam, x_raw, cutoff_hz=cutoff_hz, order=order)
    resid = x_raw - x_lpf
    std_A = np.nanstd(resid[beam == "A"], axis=0)
    return std_A > thresh_factor * np.nanmedian(std_A)


def grad_block_data(target_path: str, calib_a_csv: str, calib_a_obsid: int,
                     min_samples_per_beam: int = MIN_SAMPLES_PER_BEAM,
                     raw_noise_thresh: float = RAW_NOISY_KID_THRESH) -> dict:
    """Load a pswsc file and build the per-GRAD-block (beam A Tb, beam B
    raw) arrays that `calibrate_beamB_from_grad`/cross-validation are
    fit on. Split out on its own so cross-validation code can reuse it
    without re-deriving the fit."""
    calib_a = pd.read_csv(calib_a_csv).query("obsid == @calib_a_obsid").set_index("chan")

    da_sub = load_pswsc(target_path)
    obsid = str(da_sub.aste_obs_id)
    obj_name = str(da_sub.aste_obs_file).split("_")[0]

    state = da_sub.state.to_numpy()
    beam = da_sub.beam.to_numpy()
    chan_all = da_sub.chan.to_numpy()
    freq_all = da_sub.frequency.to_numpy()
    t_sec = utils.dt_to_seconds(da_sub)
    x_raw = da_sub.to_numpy()

    x_lpf = apply_lpf_per_beam(t_sec, beam, x_raw)

    # exclude raw-noisy KIDs (>1Hz residual std, both beams) BEFORE the
    # CV/fit even sees them -- a bad channel here would otherwise corrupt
    # the single shared frac_a/frac_b chosen by cv_select_frac for everyone
    flagged_raw = flag_noisy_kids_raw(t_sec, beam, x_raw, thresh_factor=raw_noise_thresh)

    # beam A: calibrate to Tb using its own borrowed skydip (a, b)
    common_chan, idx_obs, idx_cal = np.intersect1d(chan_all, calib_a.index.to_numpy(), return_indices=True)
    keep = ~flagged_raw[idx_obs]
    common_chan, idx_obs, idx_cal = common_chan[keep], idx_obs[keep], idx_cal[keep]
    a_A = calib_a.loc[common_chan, "a"].to_numpy()
    b_A = calib_a.loc[common_chan, "b"].to_numpy()
    freq_common = freq_all[idx_obs]

    tb_A_full = np.full((x_lpf.shape[0], len(common_chan)), np.nan)
    selA = beam == "A"
    tb_A_full[selA] = a_A[None, :] * x_lpf[np.ix_(selA, idx_obs)] + b_A[None, :]

    # GRAD-block-averaged beam A Tb / beam B raw, per channel, per block
    change = np.where(state[1:] != state[:-1])[0] + 1
    bounds = np.concatenate(([0], change, [len(state)]))
    grad_blocks = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if state[bounds[i]] == "GRAD"]

    tbA_blocks, xB_blocks = [], []
    for lo, hi in grad_blocks:
        idx = np.arange(lo, hi)
        idxA = idx[beam[idx] == "A"]
        idxB = idx[beam[idx] == "B"]
        if idxA.size < min_samples_per_beam or idxB.size < min_samples_per_beam:
            continue
        tbA_blocks.append(np.nanmean(tb_A_full[idxA], axis=0))
        xB_blocks.append(np.nanmean(x_lpf[np.ix_(idxB, idx_obs)], axis=0))
    tbA_blocks = np.array(tbA_blocks)
    xB_blocks = np.array(xB_blocks)

    return dict(obsid=obsid, obj_name=obj_name, chan=common_chan, freq=freq_common,
                a_A=a_A, b_A=b_A, tbA_blocks=tbA_blocks, xB_blocks=xB_blocks,
                state=state, beam=beam, t_sec=t_sec, x_lpf=x_lpf, idx_obs=idx_obs,
                chan_all=chan_all, tb_A_full=tb_A_full,
                chan_flagged_raw=chan_all[flagged_raw])


def calibrate_beamB_from_grad(target_path: str, calib_a_csv: str, calib_a_obsid: int,
                               min_samples_per_beam: int = MIN_SAMPLES_PER_BEAM,
                               frac_a: float = 0.0, frac_b: float = 0.0,
                               raw_noise_thresh: float = RAW_NOISY_KID_THRESH) -> dict:
    """Step 0: fit beam B's (a, b) directly against beam A's Tb, using GRAD.

    Only beam A's borrowed skydip calibration is used. Beam B's calibration
    is entirely determined by this GRAD-block regression -- no separate
    beam B skydip calibration is read or needed.

    frac_a/frac_b (default 0 = plain OLS, `scipy.stats.linregress`):
    dimensionless ridge-regularization fractions (see `_ridge_fit`) pulling
    (a_B, b_B) toward (a_A, b_A) -- i.e. penalizing (a_A-a_B)**2 and
    (b_A-b_B)**2 in the fit objective, so low-leverage channels (near-flat
    eta_atm, little real GRAD-to-GRAD variation to fit against) default
    toward assuming beam B matches beam A, instead of letting the
    slope/intercept be noise-dominated. Pick these via `cv_select_frac`
    rather than by hand.

    Returns a dict with the fitted per-channel (a_B, b_B, R2), plus the
    intermediate arrays needed to sanity-check or reuse the result
    (channel/frequency axis, tb_A array, per-beam LPF'd raw data, etc.).
    """
    d = grad_block_data(target_path, calib_a_csv, calib_a_obsid, min_samples_per_beam, raw_noise_thresh)
    return _fit_from_grad_data(d, frac_a, frac_b)


def _fit_from_grad_data(d: dict, frac_a: float, frac_b: float) -> dict:
    """Shared fit step for `calibrate_beamB_from_grad`/`step0_calibrate_beamB`,
    given an already-loaded `grad_block_data` result (avoids reloading and
    re-filtering the dems file when the caller already has it)."""
    xB_blocks, tbA_blocks, a_A, b_A = d["xB_blocks"], d["tbA_blocks"], d["a_A"], d["b_A"]
    n_blocks, n_chan = tbA_blocks.shape

    a_B = np.full(n_chan, np.nan)
    b_B = np.full(n_chan, np.nan)
    r2 = np.full(n_chan, np.nan)
    for c in range(n_chan):
        x, y = xB_blocks[:, c], tbA_blocks[:, c]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 5:
            continue
        res = linregress(x[ok], y[ok])
        r2[c] = res.rvalue ** 2  # always report the plain-OLS R^2 for diagnostics
        if frac_a == 0.0 and frac_b == 0.0:
            a_B[c], b_B[c] = res.slope, res.intercept
        else:
            a_B[c], b_B[c] = _ridge_fit(x[ok], y[ok], a_A[c], b_A[c], frac_a, frac_b)

    return dict(obsid=d["obsid"], obj_name=d["obj_name"], chan=d["chan"], freq=d["freq"],
                a_B=a_B, b_B=b_B, r2=r2, n_blocks=n_blocks,
                a_A=a_A, b_A=b_A, state=d["state"], beam=d["beam"], t_sec=d["t_sec"],
                x_lpf=d["x_lpf"], idx_obs=d["idx_obs"], chan_all=d["chan_all"], tb_A_full=d["tb_A_full"],
                chan_flagged_raw=d["chan_flagged_raw"])


def cv_select_frac(xB_blocks: np.ndarray, tbA_blocks: np.ndarray, a_A: np.ndarray, b_A: np.ndarray,
                    frac_a_grid: np.ndarray, frac_b_grid: np.ndarray) -> dict:
    """Leave-one-GRAD-block-out cross-validation grid search for
    (frac_a, frac_b), minimizing the held-out unexplained-variance
    fraction averaged EQUALLY across all channels (not summed in raw K^2,
    which would be dominated by the brightest/deep-absorption channels).

    Returns the best (frac_a, frac_b) plus the full CV objective grid,
    for inspection/plotting.
    """
    n_blocks, n_chan = tbA_blocks.shape
    var_chan = np.var(tbA_blocks, axis=0, ddof=1)
    var_chan = np.where(var_chan > 0, var_chan, np.nan)

    grid = np.full((len(frac_a_grid), len(frac_b_grid)), np.nan)
    for i, frac_a in enumerate(frac_a_grid):
        for j, frac_b in enumerate(frac_b_grid):
            sq_err = np.zeros(n_chan)
            for held_out in range(n_blocks):
                train = np.arange(n_blocks) != held_out
                a, b = _ridge_fit_vec(xB_blocks[train], tbA_blocks[train], a_A, b_A, frac_a, frac_b)
                pred = a * xB_blocks[held_out] + b
                sq_err += (tbA_blocks[held_out] - pred) ** 2
            mse_chan = sq_err / n_blocks
            grid[i, j] = np.nanmean(mse_chan / var_chan)

    i_best, j_best = np.unravel_index(np.nanargmin(grid), grid.shape)
    return dict(frac_a_grid=frac_a_grid, frac_b_grid=frac_b_grid, cv_grid=grid,
                best_frac_a=float(frac_a_grid[i_best]), best_frac_b=float(frac_b_grid[j_best]),
                best_cv_score=float(grid[i_best, j_best]))


DEFAULT_FRAC_GRID = np.logspace(-7, 2, 19)  # excludes 0.0 on purpose -- always regularize a little


def step0_calibrate_beamB(target_path: str, calib_a_csv: str, calib_a_obsid: int,
                           min_samples_per_beam: int = MIN_SAMPLES_PER_BEAM,
                           frac_a_grid: np.ndarray = DEFAULT_FRAC_GRID,
                           frac_b_grid: np.ndarray = DEFAULT_FRAC_GRID,
                           raw_noise_thresh: float = RAW_NOISY_KID_THRESH) -> dict:
    """Step 0, single entry point: calibrate beam B against beam A's Tb
    using GRAD, with the ridge-regularization strength chosen by
    leave-one-GRAD-block-out cross-validation for THIS file (never reused
    from another observation).

    Raw-noisy KIDs (`flag_noisy_kids_raw`, >`raw_noise_thresh`x median
    beam-A >1Hz residual std) are excluded BEFORE the CV grid search even
    runs, so they can't corrupt the single shared frac_a/frac_b chosen for
    every channel. This is on top of (not a replacement for)
    `faica_denoise.flag_noisy_kids`, which runs AFTER calibration on the
    FA noise variance and catches subtler, correlated-structure noise this
    raw filter can't see.

    Returns the same dict as `calibrate_beamB_from_grad`, plus `cv` (the
    full `cv_select_frac` result) and `chan_flagged_raw` (channels excluded
    by the raw pre-filter, for inspection/plotting the CV surface).
    """
    d = grad_block_data(target_path, calib_a_csv, calib_a_obsid, min_samples_per_beam, raw_noise_thresh)
    cv = cv_select_frac(d["xB_blocks"], d["tbA_blocks"], d["a_A"], d["b_A"], frac_a_grid, frac_b_grid)
    result = _fit_from_grad_data(d, cv["best_frac_a"], cv["best_frac_b"])
    result["cv"] = cv
    return result
