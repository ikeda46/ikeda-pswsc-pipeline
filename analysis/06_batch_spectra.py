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
# # Batch: run the finalized pipeline (Step 0 + LPF 1.0Hz spectrum) over all pswsc files
#
# For each pswsc file in `tau0/dems_classification.csv` (kind=="pswsc"):
# 1. Pick the nearest-in-time QC-passed skydip (from
#    `tau0/skydip_calibration_v3_lpf.csv`'s available obsids, matched via
#    `tau0/dems_catalog.csv`'s time_start -- by absolute time difference,
#    so an evening-start observation correctly compares against both the
#    previous evening's and the following early-morning's skydips, with no
#    calendar-date bias) as beam A's borrowed calibration.
# 2. Run Step 0 (balanced-GRAD-block ridge+CV beam B calibration).
# 3. Build the 1.0Hz-LPF ON-OFF spectrum only (the adopted method; raw and
#    FA+JADE are comparison-only and skipped here to keep the batch's
#    memory/time footprint down).
# 4. Save spectrum data (.npz: freq/mean/std/chan/master_id/n) + a plot
#    (.png) under `results/<obsid>_<object>/`, with the obsid AND object
#    name embedded in every filename too (not just the directory), so
#    files stay unambiguous if ever copied out flat for cross-object
#    comparison. Also saves per-channel calibration diagnostics
#    (a_A/b_A/a_B/b_B/chan/freq/master_id, CV fracs, R^2, flagged
#    channels, which skydip was used and the time gap to it).
#
# Each file is wrapped in try/except so one failure doesn't stop the batch;
# failures are logged with their error message, not silently dropped.
#
# Memory note (2026-08-10): an earlier version of this script ran raw+LPF
# (2 spectrum methods) at NPROC=24 and exhausted system memory badly enough
# to force a hard reboot. Cut to LPF-only here, and NPROC defaults much
# lower (8) -- raise it only after confirming actual peak memory use on
# this machine for a few files.
#
# Memory leak, root-caused and fixed (2026-08-12): a second, independent
# run at NPROC=24 climbed to ~200GB RSS (machine has 251GB) and was cut
# short. Diagnosed with a serial single-process probe (thread counts via
# threading.enumerate() and /proc/<pid>/status): TWO compounding
# thread-accumulation sources, both invisible to the earlier
# plt.close(fig) fix (that one was real but not the dominant cause):
#   1. `load_pswsc` (calibration.py) called decode's `dc.load.dems`
#      without `chunks=None` -- decode defaults zarr loads to
#      `chunks="auto"`, i.e. dask-backed lazy arrays, even though
#      `load_pswsc` immediately calls `.to_numpy()` and never needs
#      laziness. Every dask `.compute()` handed work to a NEW batch of
#      threads on dask's default threaded-scheduler pool that were never
#      torn down -- confirmed +20-27 Python threads per file, unbounded.
#      Fixed by passing `chunks=None` (calibration.py, `load_pswsc`).
#   2. No BLAS thread cap: numpy/scipy here link OpenBLAS, which by
#      default spins up a per-process thread pool sized to
#      `os.cpu_count()` (128 on this machine). With NPROC=24 worker
#      processes each independently doing this, that's up to 24x128=3072
#      OS threads contending for 128 cores -- severe oversubscription
#      that both wastes memory (thread stacks/thread-local buffers) and
#      caused catastrophic per-file slowdowns during the crashed run.
#      Fixed by pinning BLAS libraries to 1 thread per worker (below, via
#      env vars set BEFORE numpy is imported -- multiprocessing's
#      fork-based workers inherit whatever BLAS already initialized in
#      the parent, so this must happen at module import time, not inside
#      each worker function) and relying on process-level (NPROC)
#      parallelism instead.
# Verified with a serial 20-file probe (both fixes applied): RSS stayed
# completely flat file-to-file (deltas of a few hundred KB to ~1MB, some
# even negative) after the first file's one-time warm-up, vs. the
# unfixed version's +1.2-1.6GB per file. NPROC can likely be raised
# again given this, but hasn't been re-tested at higher NPROC since the
# fix -- confirm peak per-worker RSS empirically before doing so.

# %%
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time
import traceback
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "../src")
from ikeda_pswsc_pipeline import step0_calibrate_beamB, onoff_diff_spectrum, plot_spectrum

TAU0_DIR = "../../tau0"
CALIB_A_CSV = f"{TAU0_DIR}/skydip_calibration_v3_lpf.csv"
CATALOG_CSV = f"{TAU0_DIR}/dems_catalog.csv"
CLASSIFICATION_CSV = f"{TAU0_DIR}/dems_classification.csv"
RESULTS_DIR = "../results"

PILOT_N = 15
NPROC = 8  # lowered from 24 after an OOM crash -- see module docstring


def resolve_path(catalog_file: str) -> str:
    """catalog paths are relative to tau0/ (e.g. '../data/...'); analysis/
    scripts sit one level deeper, so need one more '../'."""
    assert catalog_file.startswith("../data/"), catalog_file
    return "../../data/" + catalog_file[len("../data/"):]


def build_skydip_lookup() -> pd.DataFrame:
    """QC-passed skydip obsids (from the calibration table actually used
    elsewhere in this pipeline) with their own observation time, sorted by
    time -- the set to search for "nearest in time" against."""
    calib_a = pd.read_csv(CALIB_A_CSV)
    skydip_obsids = calib_a["obsid"].unique()
    cat = pd.read_csv(CATALOG_CSV)
    cat_sky = cat[cat["obsid"].isin(skydip_obsids)][["obsid", "time_start"]].copy()
    cat_sky["time_start"] = pd.to_datetime(cat_sky["time_start"])
    return cat_sky.sort_values("time_start").reset_index(drop=True)


def nearest_skydip(time_start: pd.Timestamp, skydip_lookup: pd.DataFrame) -> tuple[int, float]:
    """Nearest skydip by absolute time difference (continuous timestamp
    comparison, NOT calendar-date matching) -- correctly handles
    evening-start observations against both the preceding evening's and
    the following early-morning's skydips."""
    diffs = (skydip_lookup["time_start"] - time_start).abs()
    i = diffs.idxmin()
    return int(skydip_lookup.loc[i, "obsid"]), diffs.loc[i].total_seconds() / 3600.0


def process_one(row: dict, skydip_lookup: pd.DataFrame) -> dict:
    obsid = row["obsid"]
    t0 = time.time()
    result = dict(obsid=obsid, file=row["file"], status="fail", error="")
    try:
        target_path = resolve_path(row["file"])
        pswsc_time = pd.to_datetime(row["time_start"])
        calib_a_obsid, gap_hours = nearest_skydip(pswsc_time, skydip_lookup)

        s0 = step0_calibrate_beamB(target_path, CALIB_A_CSV, calib_a_obsid)
        a_A, b_A, a_B, b_B, chan = s0["a_A"], s0["b_A"], s0["a_B"], s0["b_B"], s0["chan"]
        obj_name = s0["obj_name"]

        # master_id (decode's own KID identity field, `d2_mkid_id` --
        # distinct from `chan`, the per-file channel index, though the two
        # happen to coincide on files seen so far) for every surviving
        # channel, plus the frequencies of raw-noise-flagged channels
        # (dropped before calibration, so they need the FULL, unfiltered
        # chan/freq/master_id arrays -- already returned by step0, no need
        # to reload the file a second time just for this).
        _, idx_flagged, _ = np.intersect1d(s0["chan_all"], s0["chan_flagged_raw"], return_indices=True)
        freq_flagged = s0["freq_all"][idx_flagged]

        tag = f"{obsid}_{obj_name}"
        outdir = f"{RESULTS_DIR}/{tag}"
        os.makedirs(outdir, exist_ok=True)

        np.savez(f"{outdir}/calibration_{tag}.npz", a_A=a_A, b_A=b_A, a_B=a_B, b_B=b_B, chan=chan,
                  freq=s0["freq"], master_id=s0["master_id"], n_blocks=s0["n_blocks"], r2=s0["r2"],
                  cv_frac_a=s0["cv"]["best_frac_a"], cv_frac_b=s0["cv"]["best_frac_b"],
                  chan_flagged_raw=s0["chan_flagged_raw"], calib_a_obsid=calib_a_obsid, gap_hours=gap_hours)

        spec = onoff_diff_spectrum(target_path, a_A, b_A, a_B, b_B, chan, lpf_cutoff_hz=1.0)
        np.savez(f"{outdir}/spectrum_lpf1hz_{tag}.npz", freq=spec["freq"], mean=spec["mean"], std=spec["std"],
                  chan=spec["chan"], master_id=s0["master_id"], n=spec["n"], obsid=spec["obsid"],
                  obj_name=spec["obj_name"])
        plot_spectrum(spec, title=f"{obj_name} (obsid={obsid}): LPF 1.0Hz ({len(chan)} chan)",
                      out_path=f"{outdir}/spectrum_lpf1hz_{tag}.png", flagged_freq=freq_flagged)

        result.update(status="ok", obj_name=obj_name, calib_a_obsid=calib_a_obsid, gap_hours=gap_hours,
                       n_blocks=s0["n_blocks"], r2_median=float(np.nanmedian(s0["r2"])),
                       cv_frac_a=s0["cv"]["best_frac_a"], cv_frac_b=s0["cv"]["best_frac_b"],
                       n_flagged_raw=len(s0["chan_flagged_raw"]), n_chan_final=len(chan))
    except Exception as e:
        result.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    result["runtime_sec"] = time.time() - t0
    return result


def _worker(row, skydip_lookup):
    """Top-level wrapper so Pool can pickle/import it."""
    return process_one(row, skydip_lookup)


# %%
# Set this directly and re-run this cell + the one below -- "pilot" runs
# PILOT_N files serially (for a quick check), "full" runs all files in
# parallel (NPROC processes). Recommended order: "pilot" first, check
# results/batch_log_pilot.csv and a couple of the saved plots, THEN switch
# to "full".
MODE = "full"  # "pilot" or "full"

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    skydip_lookup = build_skydip_lookup()
    print(f"skydip lookup: {len(skydip_lookup)} QC-passed skydips available", flush=True)

    cls = pd.read_csv(CLASSIFICATION_CSV)
    cat = pd.read_csv(CATALOG_CSV)
    pswsc = cls[cls["kind"] == "pswsc"].merge(cat[["obsid", "time_start"]], on="obsid", how="left")
    print(f"total pswsc files: {len(pswsc)}", flush=True)

    if MODE == "pilot":
        rows = pswsc.sample(n=PILOT_N, random_state=0).to_dict("records")
        log_out = f"{RESULTS_DIR}/batch_log_pilot.csv"
        print(f"=== PILOT: {len(rows)} files, serial ===", flush=True)

        log_rows = []
        for row in tqdm(rows, desc="pilot"):
            r = process_one(row, skydip_lookup)
            log_rows.append(r)
            pd.DataFrame(log_rows).drop(columns=["traceback"], errors="ignore").to_csv(log_out, index=False)

    else:  # full, parallel
        rows = pswsc.to_dict("records")
        log_out = f"{RESULTS_DIR}/batch_log.csv"
        print(f"=== FULL: {len(rows)} files, {NPROC} processes ===", flush=True)

        log_rows = []
        with Pool(NPROC) as p:
            for i, r in enumerate(tqdm(p.imap_unordered(partial(_worker, skydip_lookup=skydip_lookup), rows),
                                        total=len(rows), desc="batch"), 1):
                log_rows.append(r)
                if i % 20 == 0 or i == len(rows):
                    pd.DataFrame(log_rows).drop(columns=["traceback"], errors="ignore").to_csv(log_out, index=False)

    df_log = pd.DataFrame(log_rows).drop(columns=["traceback"], errors="ignore")
    df_log.to_csv(log_out, index=False)
    n_ok = (df_log["status"] == "ok").sum()
    print(f"saved: {log_out} ({n_ok}/{len(df_log)} succeeded)", flush=True)

# %%
