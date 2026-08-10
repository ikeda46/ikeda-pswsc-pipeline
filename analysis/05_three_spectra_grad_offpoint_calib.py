# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Three spectrum-construction methods, using the GRAD off-point in-situ calibration
#
# Same as `04_three_spectra_comparison.py` (raw / LPF 1.0Hz / FA+JADE
# spectra, one independent figure per method), but the calibration stage
# is replaced: instead of Step 0 (`step0_calibrate_beamB`, ridge+CV
# regression of beam B against beam A's Tb during GRAD), both beam A and
# beam B are calibrated independently via
# `calibrate_beam_from_grad_offpoint` -- an in-situ, skydip-style PWV fit
# using only each beam's own off-point portion of the GRAD chunks (see
# `grad_skydip.py`'s module docstring).
#
# The bootstrap PWV seed is NOT the ALMA-based default: it's fit from
# beam A's EXISTING external skydip calibration applied to beam A's own
# off-point data (`pwv_seed_from_calibrated_tb`), then shared as the seed
# for BOTH beams -- found empirically (2026-08-10) to track the trusted
# external beam-A calibration far more closely than the ALMA seed.

# %%
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "../src")
from ikeda_pswsc_pipeline import (
    calibrate_beam_from_grad_offpoint, pwv_seed_from_calibrated_tb,
    onoff_diff_spectrum, fa_ica_spectrum, plot_spectrum,
)

CALIB_A_CSV = "../../tau0/skydip_calibration_v3_lpf.csv"
N_COMPONENTS_FIXED = 20

TARGETS = [
    ("ngc1068", "../../data/v20250826/dems_rm_neg/dems_20240715094140.zarr.zip", 20240715072703),
    ("mars", "../../data/v20250826/dems_rm_neg/dems_20240715115449.zarr.zip", 20240715132539),
]

# %%
for name, target_path, calib_a_obsid in TARGETS:
    print(f"=== {name} ===", flush=True)
    calib_a = pd.read_csv(CALIB_A_CSV).query("obsid == @calib_a_obsid")
    a_ext = calib_a["a"].to_numpy()
    b_ext = calib_a["b"].to_numpy()
    chan_ext = calib_a["chan"].to_numpy()

    pwv_seed = pwv_seed_from_calibrated_tb(target_path, "A", a_ext, b_ext, chan_ext)
    print(f"  shared PWV seed (from beam A's external skydip calib): {pwv_seed:.4f}mm", flush=True)

    r_A = calibrate_beam_from_grad_offpoint(target_path, "A", pwv_seed_override=pwv_seed)
    r_B = calibrate_beam_from_grad_offpoint(target_path, "B", pwv_seed_override=pwv_seed)
    print(f"  beam A: n_chan={len(r_A['chan'])}  R2 median={np.nanmedian(r_A['r2']):.4f}", flush=True)
    print(f"  beam B: n_chan={len(r_B['chan'])}  R2 median={np.nanmedian(r_B['r2']):.4f}", flush=True)

    chan, i_A, i_B = np.intersect1d(r_A["chan"], r_B["chan"], return_indices=True)
    a_A, b_A = r_A["a"][i_A], r_A["b"][i_A]
    a_B, b_B = r_B["a"][i_B], r_B["b"][i_B]
    print(f"  common channels: {len(chan)}", flush=True)

    spec_raw = onoff_diff_spectrum(target_path, a_A, b_A, a_B, b_B, chan, lpf_cutoff_hz=None)
    print("  raw spectrum: done", flush=True)
    spec_lpf = onoff_diff_spectrum(target_path, a_A, b_A, a_B, b_B, chan, lpf_cutoff_hz=1.0)
    print("  LPF 1.0Hz spectrum: done", flush=True)
    spec_fa = fa_ica_spectrum(target_path, a_A, b_A, a_B, b_B, chan, n_components=N_COMPONENTS_FIXED)
    print("  FA(m=20)+JADE spectrum: done", flush=True)

    for spec, method, title in [
        (spec_raw, "raw", "raw (no LPF)"),
        (spec_lpf, "lpf1hz", "LPF 1.0Hz"),
        (spec_fa, "faica", f"FA(m={N_COMPONENTS_FIXED})+JADE reconstruction"),
    ]:
        out_path = f"05_spectrum_{method}_{name}_{spec['obsid']}.png"
        plot_spectrum(spec, title=f"{spec['obj_name']} (obsid={spec['obsid']}): {title} "
                                   f"({len(chan)} chan) -- GRAD off-point calibration",
                      out_path=out_path)
        print(f"  saved: {out_path}", flush=True)

# %%
