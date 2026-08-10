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
# # Three spectrum-construction methods: raw / LPF 1.0Hz / FA+JADE
#
# Pipeline reads as three independent stages: calibration -> spectrum
# construction -> display. Each of the three methods gets its own
# standalone figure (`plotting.plot_spectrum`), not a shared subplot grid
# -- display is a single reusable function, called once per method,
# instead of layout code tied to "there are exactly 3 of these".
#
# Raw-noisy KIDs are excluded (beam-A >1Hz-residual std, 10x median,
# `calibration.flag_noisy_kids_raw`) BEFORE Step 0's ridge+CV calibration
# even runs -- a KID flagged here is dropped from this observation
# entirely, for all three spectra below. Then Step 0 (ridge + leave-one-
# GRAD-block-out CV) determines beam B's (a, b) against beam A's Tb. The
# three spectra below all use that same calibration and channel set, just
# built via different post-calibration processing of the timestream.

# %%
import sys

import numpy as np

sys.path.insert(0, "../src")
from ikeda_pswsc_pipeline import (
    step0_calibrate_beamB, onoff_diff_spectrum, fa_ica_spectrum, plot_spectrum, load_pswsc,
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
    s0 = step0_calibrate_beamB(target_path, CALIB_A_CSV, calib_a_obsid)
    a_A, b_A, a_B, b_B, chan = s0["a_A"], s0["b_A"], s0["a_B"], s0["b_B"], s0["chan"]
    print(f"  raw-flagged excluded: {s0['chan_flagged_raw']} -> {len(chan)} channels remain", flush=True)
    print(f"  CV: frac_a={s0['cv']['best_frac_a']:.2e} frac_b={s0['cv']['best_frac_b']:.2e}", flush=True)

    da_full = load_pswsc(target_path)
    _, idx_flagged, _ = np.intersect1d(da_full.chan.to_numpy(), s0["chan_flagged_raw"], return_indices=True)
    freq_flagged = da_full.frequency.to_numpy()[idx_flagged]

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
        out_path = f"04_spectrum_{method}_{name}_{s0['obsid']}.png"
        plot_spectrum(spec, title=f"{spec['obj_name']} (obsid={spec['obsid']}): {title} ({len(chan)} chan)",
                      out_path=out_path, flagged_freq=freq_flagged)
        print(f"  saved: {out_path}", flush=True)

# %%
