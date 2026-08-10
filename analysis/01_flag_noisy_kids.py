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
# # Step 0 -> Step 1-4: full pipeline using only the finalized src/ functions
#
# 1. Step 0: calibrate beam B against beam A directly from GRAD (ridge +
#    leave-one-block-out CV for the shrinkage fractions).
# 2. Fit FA (fixed m=20; MDL/AIC model selection was tried and dropped --
#    both kept decreasing well past any computationally practical m for
#    JADE) + JADE (no whitening) on each beam's own full ON+OFF signal.
# 3. Flag noisy KIDs from the FA noise std (>10x median, in BOTH beams
#    independently) -- adopted over IC-removal-and-reconstruct, which had
#    the side effect of increasing noise in other channels.
# 4. Plot the plain (non-FA-denoised) ON-OFF spectrum with flagged KIDs
#    marked.

# %%
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "../src")
from ikeda_pswsc_pipeline import (
    step0_calibrate_beamB, apply_lpf_per_beam, load_pswsc, fit_faica,
    flag_noisy_kids, onoff_diff_spectrum,
)
from d24_tools import utils as d24_utils

CALIB_A_CSV = "../../tau0/skydip_calibration_v3_lpf.csv"
TARGET_PATH = "../../data/v20250826/dems_rm_neg/dems_20240715094140.zarr.zip"  # ngc1068
CALIB_A_OBSID = 20240715072703

N_COMPONENTS_FIXED = 20
NOISY_KID_THRESH = 10.0

# %%
# --- Step 0: beam B calibration (ridge + CV) --------------------------------
s0 = step0_calibrate_beamB(TARGET_PATH, CALIB_A_CSV, CALIB_A_OBSID)
print(f"obsid={s0['obsid']}, object={s0['obj_name']}, "
      f"CV best frac_a={s0['cv']['best_frac_a']:.2e}, frac_b={s0['cv']['best_frac_b']:.2e}")
a_A, b_A, a_B, b_B, chan_common = s0["a_A"], s0["b_A"], s0["a_B"], s0["b_B"], s0["chan"]

# %%
# --- load full file, per-beam 1.0Hz LPF, ON+OFF only ------------------------
da_sub = load_pswsc(TARGET_PATH)
state = da_sub.state.to_numpy()
beam = da_sub.beam.to_numpy()
chan_all = da_sub.chan.to_numpy()
freq_all = da_sub.frequency.to_numpy()
t_sec = d24_utils.dt_to_seconds(da_sub)
x_raw = da_sub.to_numpy()

x_lpf = apply_lpf_per_beam(t_sec, beam, x_raw, cutoff_hz=1.0)
_, idx_obs, _ = np.intersect1d(chan_all, chan_common, return_indices=True)
freq_common = freq_all[idx_obs]
onoff_state_mask = (state == "ON") | (state == "OFF")

# %%
# --- fixed-m=20 FA+JADE per beam, keep only the noise_variance_ we need ----
noise_variance = {}
for bm, (a_bm, b_bm) in [("A", (a_A, b_A)), ("B", (a_B, b_B))]:
    print(f"=== beam {bm}: fitting FA (m={N_COMPONENTS_FIXED}) + JADE ===")
    sel = (beam == bm) & onoff_state_mask
    tb_bm = a_bm[None, :] * x_lpf[np.ix_(sel, idx_obs)] + b_bm[None, :]
    fit = fit_faica(tb_bm, n_components=N_COMPONENTS_FIXED)
    noise_variance[bm] = fit["fa"].noise_variance_
    print(f"  done, m={fit['m']}")

# %%
flagged = flag_noisy_kids(noise_variance["A"], noise_variance["B"], thresh_factor=NOISY_KID_THRESH)
print(f"flagged {flagged.sum()}/{len(flagged)} KIDs (threshold={NOISY_KID_THRESH}x median, both beams):")
for f in freq_common[flagged]:
    print(f"  {f:.2f}GHz")

# %%
# --- plain (non-FA-denoised) ON-OFF spectrum, 1.0Hz LPF ---------------------
spec = onoff_diff_spectrum(TARGET_PATH, a_A, b_A, a_B, b_B, chan_common, lpf_cutoff_hz=1.0)

order = np.argsort(freq_common)
fig, ax = plt.subplots(figsize=(11, 5))
ax.errorbar(spec["freq"][order], spec["mean"][order], yerr=spec["std"][order],
            fmt="o", ms=3, lw=0.8, capsize=2, color="tab:blue", ecolor="tab:blue", alpha=0.8,
            label="tb_on - tb_off")
ax.plot(freq_common[flagged], spec["mean"][flagged], "x", ms=10, mew=2, color="red",
        label=f"flagged ({NOISY_KID_THRESH:.0f}x median std, both beams)")
ax.axhline(0, color="k", lw=0.5)
ax.set(xlabel="Frequency [GHz]", ylabel="tb_on - tb_off [K]",
       title=f"{spec['obj_name']} (obsid={spec['obsid']}): step0->step1-4 pipeline "
             f"(noisy-KID flags from FA m={N_COMPONENTS_FIXED} noise std)")
ax.legend(fontsize=8)
fig.tight_layout()
out_path = f"01_flagged_spectrum_{s0['obsid']}.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved: {out_path}")

# %%
