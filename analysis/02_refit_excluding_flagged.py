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
# # Step 1-4 round 2: exclude flagged KIDs, refit FA+JADE, check criterion 4 again
#
# Round 1 flagged 265.52/321.51 GHz (>10x median FA noise std, both beams).
# Here those 2 channels are dropped from the channel set BEFORE fitting (not
# just marked after), FA(m=20)+JADE is refit on the remaining 248 channels,
# and criterion 4 (concentrated loading) is checked again -- does removing
# the two worst channels reveal a THIRD channel now dominating some IC?
#
# Since the excluded channels have no spectrum data point in this reduced
# run, their frequency positions are marked with an x placed ABOVE the plot
# (fixed height), not at a (freq, value) data point.

# %%
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "../src")
from ikeda_pswsc_pipeline import (
    step0_calibrate_beamB, apply_lpf_per_beam, load_pswsc, fit_faica,
    ic_diagnostics, flag_noisy_kids, onoff_diff_spectrum,
)
from d24_tools import utils as d24_utils

CALIB_A_CSV = "../../tau0/skydip_calibration_v3_lpf.csv"
TARGET_PATH = "../../data/v20250826/dems_rm_neg/dems_20240715094140.zarr.zip"  # ngc1068
CALIB_A_OBSID = 20240715072703

N_COMPONENTS_FIXED = 20
NOISY_KID_THRESH = 10.0

# %%
# --- Step 0 + round-1 flagging (reuse cached noise_variance from round 1) --
s0 = step0_calibrate_beamB(TARGET_PATH, CALIB_A_CSV, CALIB_A_OBSID)
a_A, b_A, a_B, b_B, chan_common = s0["a_A"], s0["b_A"], s0["a_B"], s0["b_B"], s0["chan"]

npz_A = np.load(f"../tmp/step1-4_fit_beamA_{s0['obsid']}.npz")
npz_B = np.load(f"../tmp/step1-4_fit_beamB_{s0['obsid']}.npz")
assert np.array_equal(npz_A["chan"], chan_common)
flagged_round1 = flag_noisy_kids(npz_A["noise_variance"], npz_B["noise_variance"], NOISY_KID_THRESH)
freq_full = npz_A["freq"]
print(f"round-1 flagged: {freq_full[flagged_round1]}")

# --- reduced channel set, excluding the flagged KIDs ------------------------
keep = ~flagged_round1
chan_reduced = chan_common[keep]
a_A_r, b_A_r = a_A[keep], b_A[keep]
a_B_r, b_B_r = a_B[keep], b_B[keep]
print(f"reduced channel set: {keep.sum()}/{len(keep)} channels")

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
_, idx_obs, _ = np.intersect1d(chan_all, chan_reduced, return_indices=True)
freq_reduced = freq_all[idx_obs]
onoff_state_mask = (state == "ON") | (state == "OFF")

# %%
# --- refit FA(m=20)+JADE on the reduced (248-chan) set, per beam -----------
results = {}
for bm, (a_bm, b_bm) in [("A", (a_A_r, b_A_r)), ("B", (a_B_r, b_B_r))]:
    print(f"\n=== beam {bm} (reduced, {keep.sum()} chan) ===")
    sel = (beam == bm) & onoff_state_mask
    tb_bm = a_bm[None, :] * x_lpf[np.ix_(sel, idx_obs)] + b_bm[None, :]
    state_bm = state[sel]
    t_bm = t_sec[sel]
    fs_bm = 1.0 / np.median(np.diff(t_bm))
    onoff_ref = np.where(state_bm == "ON", 1.0, -1.0)

    fit = fit_faica(tb_bm, n_components=N_COMPONENTS_FIXED)
    print(f"fixed m={fit['m']}")

    diags = ic_diagnostics(fit["S"], fit["A_mix"], onoff_ref, fs=fs_bm)
    df = pd.DataFrame(diags)
    df["flag_c1_low_corr"] = df["abs_corr"] < 0.5
    df["flag_c2_spike"] = df["frac_energy_top5pct"] > 0.9
    df["flag_c3_high_freq"] = df["high_freq_frac"] > 0.5
    df["flag_c4_concentrated"] = df["n_chan_eff"] < 5
    print(df[["k", "abs_corr", "n_chan_eff", "flag_c1_low_corr", "flag_c2_spike",
              "flag_c3_high_freq", "flag_c4_concentrated"]].to_string(index=False))

    results[bm] = dict(fit=fit, diags=df)
    np.savez(f"02_fit_beam{bm}_{s0['obsid']}.npz", noise_variance=fit["fa"].noise_variance_)

# %%
# --- criterion-4 loadings this round, if any --------------------------------
for bm, r in results.items():
    df = r["diags"]
    c4 = df.index[df["flag_c4_concentrated"]].tolist()
    print(f"beam {bm}: criterion-4 ICs this round: {c4}")
    if not c4:
        continue
    A_mix = r["fit"]["A_mix"]
    order = np.argsort(freq_reduced)
    fig, axes = plt.subplots(len(c4), 1, figsize=(12, 3 * len(c4)), squeeze=False)
    for i, k in enumerate(c4):
        ax = axes[i, 0]
        ax.plot(freq_reduced[order], A_mix[order, k], "o-", ms=3, lw=0.6, color="tab:green")
        ax.axhline(0, color="k", lw=0.5)
        row = df.loc[k]
        ax.set_ylabel(f"IC{k} loading")
        ax.set_title(f"n_chan_eff={row['n_chan_eff']:.2f}, |corr|={row['abs_corr']:.2f}", fontsize=8, loc="right")
    axes[-1, 0].set_xlabel("Frequency [GHz]")
    fig.suptitle(f"{s0['obj_name']} beam {bm}: round-2 criterion-4 loadings (after excluding round-1 flags)", y=1.0)
    fig.tight_layout()
    out_path = f"02_c4_loadings_beam{bm}_{s0['obsid']}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")

# %%
# --- round-2 flagging from the refit noise variance -------------------------
flagged_round2 = flag_noisy_kids(results["A"]["fit"]["fa"].noise_variance_,
                                  results["B"]["fit"]["fa"].noise_variance_, NOISY_KID_THRESH)
print(f"round-2 flagged (of the reduced set): {freq_reduced[flagged_round2]}")

# %%
# --- spectrum on the reduced channel set; round-1-excluded KIDs marked
#     ABOVE the plot (no data point exists for them here) -------------------
spec = onoff_diff_spectrum(TARGET_PATH, a_A_r, b_A_r, a_B_r, b_B_r, chan_reduced, lpf_cutoff_hz=1.0)

order = np.argsort(freq_reduced)
fig, ax = plt.subplots(figsize=(11, 5))
ax.errorbar(spec["freq"][order], spec["mean"][order], yerr=spec["std"][order],
            fmt="o", ms=3, lw=0.8, capsize=2, color="tab:blue", ecolor="tab:blue", alpha=0.8,
            label="tb_on - tb_off (248 chan)")
ymax = np.nanmax(spec["mean"][order] + spec["std"][order])
y_marker = ymax * 1.15
ax.plot(freq_full[flagged_round1], np.full(flagged_round1.sum(), y_marker), "x", ms=10, mew=2,
        color="red", clip_on=False, label="excluded (round-1 flagged, no data point here)")
if flagged_round2.sum():
    ax.plot(freq_reduced[flagged_round2], spec["mean"][flagged_round2], "x", ms=10, mew=2,
            color="darkorange", label="round-2 flagged")
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
