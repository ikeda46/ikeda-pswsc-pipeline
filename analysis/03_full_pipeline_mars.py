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
# # Full pipeline (Step 0 -> Step 1-4, round 1 + round 2) for Mars
#
# Same procedure validated on NGC1068:
# 1. Step 0: beam B calibration from GRAD (ridge + CV).
# 2. Round 1: FA(m=20)+JADE per beam on all channels; flag noisy KIDs
#    (>10x median FA noise std, both beams).
# 3. Round 2: exclude round-1-flagged KIDs, refit FA+JADE, check whether a
#    new channel emerges as criterion-4 (concentrated loading) dominant.
# 4. Final spectrum (reduced channel set) with excluded KIDs marked as
#    small x's ON the Tb=0 line (no data point exists for them).
#
# Mars is a harder case: its GRAD-block affine R^2 was much worse than
# NGC1068's in earlier (2026-08-09) work, so Step 0's ridge+CV may behave
# differently here.

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
TARGET_PATH = "../../data/v20250826/dems_rm_neg/dems_20240715115449.zarr.zip"  # mars
CALIB_A_OBSID = 20240715132539

N_COMPONENTS_FIXED = 20
NOISY_KID_THRESH = 10.0

# %%
# --- Step 0 -------------------------------------------------------------
s0 = step0_calibrate_beamB(TARGET_PATH, CALIB_A_CSV, CALIB_A_OBSID)
print(f"obsid={s0['obsid']}, object={s0['obj_name']}, "
      f"CV best frac_a={s0['cv']['best_frac_a']:.2e}, frac_b={s0['cv']['best_frac_b']:.2e}")
a_A, b_A, a_B, b_B, chan_common = s0["a_A"], s0["b_A"], s0["a_B"], s0["b_B"], s0["chan"]

da_sub = load_pswsc(TARGET_PATH)
state = da_sub.state.to_numpy()
beam = da_sub.beam.to_numpy()
chan_all = da_sub.chan.to_numpy()
freq_all = da_sub.frequency.to_numpy()
t_sec = d24_utils.dt_to_seconds(da_sub)
x_raw = da_sub.to_numpy()
x_lpf = apply_lpf_per_beam(t_sec, beam, x_raw, cutoff_hz=1.0)
onoff_state_mask = (state == "ON") | (state == "OFF")

_, idx_obs_full, _ = np.intersect1d(chan_all, chan_common, return_indices=True)
freq_full = freq_all[idx_obs_full]

# %%
# --- round 1: FA+JADE on all channels, per beam --------------------------
round1 = {}
for bm, (a_bm, b_bm) in [("A", (a_A, b_A)), ("B", (a_B, b_B))]:
    print(f"=== round1: beam {bm} ({len(chan_common)} chan) ===")
    sel = (beam == bm) & onoff_state_mask
    tb_bm = a_bm[None, :] * x_lpf[np.ix_(sel, idx_obs_full)] + b_bm[None, :]
    state_bm = state[sel]
    t_bm = t_sec[sel]
    fs_bm = 1.0 / np.median(np.diff(t_bm))
    onoff_ref = np.where(state_bm == "ON", 1.0, -1.0)
    fit = fit_faica(tb_bm, n_components=N_COMPONENTS_FIXED)
    print(f"  done, m={fit['m']}")

    diags = ic_diagnostics(fit["S"], fit["A_mix"], onoff_ref, fs=fs_bm)
    df = pd.DataFrame(diags)
    df["flag_c1_low_corr"] = df["abs_corr"] < 0.5
    df["flag_c2_spike"] = df["frac_energy_top5pct"] > 0.9
    df["flag_c3_high_freq"] = df["high_freq_frac"] > 0.5
    df["flag_c4_concentrated"] = df["n_chan_eff"] < 5
    c4 = df.index[df["flag_c4_concentrated"]].tolist()
    print(f"  round1 beam {bm}: criterion-4 ICs: {c4}, "
          f"criterion-2 ICs: {df.index[df['flag_c2_spike']].tolist()}")
    round1[bm] = dict(fit=fit, diags=df)

flagged_round1 = flag_noisy_kids(round1["A"]["fit"]["fa"].noise_variance_,
                                  round1["B"]["fit"]["fa"].noise_variance_, NOISY_KID_THRESH)
print(f"round-1 flagged: {freq_full[flagged_round1]}")

# %%
# --- round 2: exclude round-1-flagged KIDs, refit ------------------------
keep = ~flagged_round1
chan_reduced = chan_common[keep]
a_A_r, b_A_r, a_B_r, b_B_r = a_A[keep], b_A[keep], a_B[keep], b_B[keep]
_, idx_obs_r, _ = np.intersect1d(chan_all, chan_reduced, return_indices=True)
freq_reduced = freq_all[idx_obs_r]
print(f"reduced channel set: {keep.sum()}/{len(keep)} channels")

round2 = {}
for bm, (a_bm, b_bm) in [("A", (a_A_r, b_A_r)), ("B", (a_B_r, b_B_r))]:
    print(f"=== round2: beam {bm} ({keep.sum()} chan) ===")
    sel = (beam == bm) & onoff_state_mask
    tb_bm = a_bm[None, :] * x_lpf[np.ix_(sel, idx_obs_r)] + b_bm[None, :]
    state_bm = state[sel]
    t_bm = t_sec[sel]
    fs_bm = 1.0 / np.median(np.diff(t_bm))
    onoff_ref = np.where(state_bm == "ON", 1.0, -1.0)
    fit = fit_faica(tb_bm, n_components=N_COMPONENTS_FIXED)
    print(f"  done, m={fit['m']}")

    diags = ic_diagnostics(fit["S"], fit["A_mix"], onoff_ref, fs=fs_bm)
    df = pd.DataFrame(diags)
    df["flag_c4_concentrated"] = df["n_chan_eff"] < 5
    c4 = df.index[df["flag_c4_concentrated"]].tolist()
    print(f"  round2 beam {bm}: criterion-4 ICs: {c4}")
    round2[bm] = dict(fit=fit, diags=df)

    if c4:
        A_mix = fit["A_mix"]
        order = np.argsort(freq_reduced)
        fig, axes = plt.subplots(len(c4), 1, figsize=(12, 3 * len(c4)), squeeze=False)
        for i, k in enumerate(c4):
            ax = axes[i, 0]
            ax.plot(freq_reduced[order], A_mix[order, k], "o-", ms=3, lw=0.6, color="tab:green")
            ax.axhline(0, color="k", lw=0.5)
            row = df.loc[k]
            ax.set_ylabel(f"IC{k} loading")
            ax.set_title(f"n_chan_eff={row['n_chan_eff']:.2f}", fontsize=8, loc="right")
        axes[-1, 0].set_xlabel("Frequency [GHz]")
        fig.suptitle(f"{s0['obj_name']} beam {bm}: round-2 criterion-4 loadings", y=1.0)
        fig.tight_layout()
        out_path = f"03_c4_loadings_beam{bm}_{s0['obsid']}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {out_path}")

flagged_round2 = flag_noisy_kids(round2["A"]["fit"]["fa"].noise_variance_,
                                  round2["B"]["fit"]["fa"].noise_variance_, NOISY_KID_THRESH)
print(f"round-2 flagged (of the reduced set): {freq_reduced[flagged_round2]}")

# %%
# --- final spectrum, excluded KIDs as small x's on the Tb=0 line ---------
spec = onoff_diff_spectrum(TARGET_PATH, a_A_r, b_A_r, a_B_r, b_B_r, chan_reduced, lpf_cutoff_hz=1.0)

order = np.argsort(freq_reduced)
fig, ax = plt.subplots(figsize=(11, 5))
ax.errorbar(spec["freq"][order], spec["mean"][order], yerr=spec["std"][order],
            fmt="o", ms=3, lw=0.8, capsize=2, color="tab:blue", ecolor="tab:blue", alpha=0.8,
            label=f"tb_on - tb_off ({keep.sum()} chan)")
ax.plot(freq_full[flagged_round1], np.zeros(flagged_round1.sum()), "x", ms=5, mew=1.2,
        color="red", label="excluded (round-1 flagged)")
if flagged_round2.sum():
    ax.plot(freq_reduced[flagged_round2], np.zeros(flagged_round2.sum()), "x", ms=5, mew=1.2,
            color="darkorange", label="round-2 flagged")
ax.axhline(0, color="k", lw=0.5)
ax.set(xlabel="Frequency [GHz]", ylabel="tb_on - tb_off [K]",
       title=f"{spec['obj_name']} (obsid={spec['obsid']}): spectrum with flagged KIDs excluded")
ax.legend(fontsize=8)
fig.tight_layout()
out_path = f"03_spectrum_mars_{s0['obsid']}.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved: {out_path}")

# %%
