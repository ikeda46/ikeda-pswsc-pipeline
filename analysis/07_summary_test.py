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
# # Prototype: combined 3-panel summary (spectrum / PWV(t) / T*) for NGC1068 + Mars
#
# Test of the new `pwv.py` module before wiring it into
# `06_batch_spectra.py`'s full-catalog batch:
# 1. ON-OFF diff spectrum, no error bars.
# 2. OFF-point PWV(t), 60s sliding window / 20s step, model eta_atm.
# 3. T* = spectrum / (mean eta_atm per channel), channels with mean
#    eta_atm < 0.2 flagged (red x) rather than plotted.
# All three combined into one PNG per file. Also measures RSS/time for
# the new PWV-estimation step specifically (on top of the existing
# Step0 + spectrum steps), per the memory-leak-fix follow-up requested
# alongside this work.

# %%
import gc
import sys
import time

import numpy as np
import psutil

sys.path.insert(0, "../src")
from ikeda_pswsc_pipeline import (
    step0_calibrate_beamB, onoff_diff_spectrum, plot_spectrum,
    estimate_pwv_timeseries, mean_eta_atm_per_chan, tstar_spectrum, plot_summary,
)

proc = psutil.Process()


def rss_mb():
    return proc.memory_info().rss / 1e6


CALIB_A_CSV = "../../tau0/skydip_calibration_v3_lpf.csv"
TARGETS = {
    "ngc1068": dict(path="../../data/v20250826/dems_rm_neg/dems_20240715094140.zarr.zip",
                     calib_a_obsid=20240715072703),
    "mars": dict(path="../../data/v20250826/dems_rm_neg/dems_20240715115449.zarr.zip",
                 calib_a_obsid=20240715132539),
}

# %%
for name, cfg in TARGETS.items():
    print(f"=== {name} ===", flush=True)
    gc.collect()
    rss0 = rss_mb()

    t0 = time.time()
    s0 = step0_calibrate_beamB(cfg["path"], CALIB_A_CSV, cfg["calib_a_obsid"])
    a_A, b_A, a_B, b_B, chan = s0["a_A"], s0["b_A"], s0["a_B"], s0["b_B"], s0["chan"]
    t_step0 = time.time() - t0
    rss_step0 = rss_mb()

    t0 = time.time()
    spec = onoff_diff_spectrum(cfg["path"], a_A, b_A, a_B, b_B, chan, lpf_cutoff_hz=1.0)
    t_spec = time.time() - t0
    rss_spec = rss_mb()

    t0 = time.time()
    df_pwv = estimate_pwv_timeseries(cfg["path"], a_A, b_A, a_B, b_B, chan)
    t_pwv = time.time() - t0
    rss_pwv = rss_mb()

    eta_result = mean_eta_atm_per_chan(df_pwv, chan, s0["freq"])
    tstar = tstar_spectrum(spec, eta_result)

    print(f"  obsid={s0['obsid']} object={s0['obj_name']}", flush=True)
    print(f"  timing: step0={t_step0:.1f}s spectrum={t_spec:.1f}s pwv={t_pwv:.1f}s", flush=True)
    print(f"  RSS: start={rss0:.0f}MB after_step0={rss_step0:.0f}MB after_spec={rss_spec:.0f}MB "
          f"after_pwv={rss_pwv:.0f}MB (delta_pwv={rss_pwv - rss_spec:.0f}MB)", flush=True)
    print(f"  pwv windows: {len(df_pwv)}, pwv range: {df_pwv['pwv'].min():.3f}-{df_pwv['pwv'].max():.3f}mm, "
          f"mean={df_pwv['pwv'].mean():.3f}mm", flush=True)
    print(f"  eta_atm-fit channels: {len(eta_result['chan'])}, "
          f"flagged (eta_atm<{eta_result['eta_min']}): {eta_result['flagged'].sum()}", flush=True)

    _, idx_flagged, _ = np.intersect1d(s0["chan_all"], s0["chan_flagged_raw"], return_indices=True)
    freq_flagged_raw = s0["freq_all"][idx_flagged]

    plot_summary(spec, df_pwv, tstar, title=f"{s0['obj_name']} (obsid={s0['obsid']})",
                 out_path=f"07_summary_{name}_{s0['obsid']}.png", flagged_freq_raw=freq_flagged_raw)
    print(f"  saved: 07_summary_{name}_{s0['obsid']}.png", flush=True)

    np.savez(f"07_pwv_timeseries_{name}_{s0['obsid']}.npz",
             t=df_pwv["t"].to_numpy(), pwv=df_pwv["pwv"].to_numpy(),
             el=df_pwv["el"].to_numpy(), t_amb=df_pwv["t_amb"].to_numpy(), n=df_pwv["n"].to_numpy())

    del s0, spec, df_pwv, eta_result, tstar
    gc.collect()
    print(f"  RSS after cleanup: {rss_mb():.0f}MB", flush=True)

# %%
