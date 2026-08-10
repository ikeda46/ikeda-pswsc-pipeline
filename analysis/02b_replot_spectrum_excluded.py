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
# # Replot 02: excluded-KID markers ON the Tb=0 line, smaller

# %%
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "../src")
from ikeda_pswsc_pipeline import step0_calibrate_beamB, flag_noisy_kids, onoff_diff_spectrum

CALIB_A_CSV = "../../tau0/skydip_calibration_v3_lpf.csv"
TARGET_PATH = "../../data/v20250826/dems_rm_neg/dems_20240715094140.zarr.zip"  # ngc1068
CALIB_A_OBSID = 20240715072703
NOISY_KID_THRESH = 10.0

# %%
s0 = step0_calibrate_beamB(TARGET_PATH, CALIB_A_CSV, CALIB_A_OBSID)
a_A, b_A, a_B, b_B, chan_common = s0["a_A"], s0["b_A"], s0["a_B"], s0["b_B"], s0["chan"]

npz_A = np.load(f"../tmp/step1-4_fit_beamA_{s0['obsid']}.npz")
npz_B = np.load(f"../tmp/step1-4_fit_beamB_{s0['obsid']}.npz")
flagged_round1 = flag_noisy_kids(npz_A["noise_variance"], npz_B["noise_variance"], NOISY_KID_THRESH)
freq_full = npz_A["freq"]

keep = ~flagged_round1
chan_reduced = chan_common[keep]
a_A_r, b_A_r, a_B_r, b_B_r = a_A[keep], b_A[keep], a_B[keep], b_B[keep]

# %%
spec = onoff_diff_spectrum(TARGET_PATH, a_A_r, b_A_r, a_B_r, b_B_r, chan_reduced, lpf_cutoff_hz=1.0)
freq_reduced = spec["freq"]

# %%
order = np.argsort(freq_reduced)
fig, ax = plt.subplots(figsize=(11, 5))
ax.errorbar(spec["freq"][order], spec["mean"][order], yerr=spec["std"][order],
            fmt="o", ms=3, lw=0.8, capsize=2, color="tab:blue", ecolor="tab:blue", alpha=0.8,
            label="tb_on - tb_off (248 chan)")
ax.plot(freq_full[flagged_round1], np.zeros(flagged_round1.sum()), "x", ms=5, mew=1.2,
        color="red", label="excluded (round-1 flagged, no data point here)")
ax.axhline(0, color="k", lw=0.5)
ax.set(xlabel="Frequency [GHz]", ylabel="tb_on - tb_off [K]",
       title=f"{spec['obj_name']} (obsid={spec['obsid']}): spectrum with round-1-flagged KIDs "
             f"excluded before fitting")
ax.legend(fontsize=8)
fig.tight_layout()
out_path = f"02_spectrum_excluded_{s0['obsid']}.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved: {out_path}")

# %%
