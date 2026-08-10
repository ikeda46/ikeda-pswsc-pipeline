# ikeda-pswsc-pipeline

Calibration and ON-OFF spectrum extraction pipeline for DESHIMA/ASTE
position-switching (**pswsc**) observations.

## Background

pswsc observations nod the telescope between an ON-source and an OFF-source
(blank sky) pointing every ~60s, while a 4-mirror wheel keeps two beams
(A/B) alternating every sample. At any instant, whichever beam is
on-source, the *other* beam is simultaneously looking at blank sky — so a
single file already contains a genuinely simultaneous on/off pair,
provided both beams can be calibrated into brightness temperature (Tb) on
a common, trustworthy scale.

This pipeline does that calibration (**Step 0**) and then builds and
compares several ways of turning the resulting Tb time series into a
per-channel ON−OFF spectrum.

## Pipeline stages

The code is organized so a script reads as three stages: **calibration →
spectrum construction → display**.

### 1. Calibration (`calibration.py`)

- Beam A is calibrated from its own borrowed skydip (external; not part of
  this repo).
- **Raw-noisy-KID pre-filter** (`flag_noisy_kids_raw`): channels whose
  >1.0Hz raw-signal residual noise (beam A) exceeds 10x the median are
  dropped *before* calibration — otherwise a single pathological channel
  can corrupt the shared regularization strength chosen for every channel
  in the next step. A KID flagged here is excluded from the observation
  entirely.
- **Step 0** (`step0_calibrate_beamB`): beam B's calibration is fit
  *directly* against beam A's Tb during `GRAD` — the brief antenna
  azimuth-slew transition between the ON and OFF nod positions, where both
  beams' samples occur at essentially the same time and pointing. The fit
  is ridge-regularized toward beam A's own (a, b), with the shrinkage
  strength chosen per-file by leave-one-GRAD-block-out cross-validation
  (`cv_select_frac`). This makes beam B's own independent skydip
  calibration unnecessary.

### 2. Spectrum construction (`spectrum.py`)

Three independent methods, all built from the *same* Step 0 calibration
and channel set, and all using simultaneous, spline-time-aligned ON−OFF
differencing (no atmospheric model anywhere):

| function | processing before differencing |
|---|---|
| `onoff_diff_spectrum(..., lpf_cutoff_hz=None)` | raw, per-sample Tb |
| `onoff_diff_spectrum(..., lpf_cutoff_hz=1.0)` | 1.0Hz low-pass filter (per beam, applied *after* splitting by beam — filtering before splitting blends the two beams' fast-alternating samples together) |
| `fa_ica_spectrum` | Factor Analysis (m=20) + JADE rotation, reconstructed from all m components (drops each channel's own idiosyncratic noise, keeps everything explained by shared factors) |

### 3. Noisy-KID flagging, post-calibration (`faica_denoise.py`)

A second, complementary QC layer: `flag_noisy_kids` fits FA(m=20)+JADE
separately per beam on that beam's calibrated Tb, and flags a channel
whose FA noise std exceeds 10x the median *in both beams independently*.
This catches subtler, correlated-noise-structure problems that the raw
pre-filter (stage 1) can't see, since it can only look at data before any
factor model exists.

### 4. Display (`plotting.py`)

`plot_spectrum(spec, title=..., out_path=...)` — one reusable function,
called once per spectrum, each producing its own standalone figure.

## Usage

```python
from ikeda_pswsc_pipeline import (
    step0_calibrate_beamB, onoff_diff_spectrum, fa_ica_spectrum, plot_spectrum,
)

s0 = step0_calibrate_beamB(target_path, calib_a_csv, calib_a_obsid)
a_A, b_A, a_B, b_B, chan = s0["a_A"], s0["b_A"], s0["a_B"], s0["b_B"], s0["chan"]

spec = onoff_diff_spectrum(target_path, a_A, b_A, a_B, b_B, chan, lpf_cutoff_hz=1.0)
plot_spectrum(spec, title="my target", out_path="spectrum.png")
```

See `analysis/04_three_spectra_comparison.py` for a full worked example
(all three spectrum methods, run for two targets).

## Installation

```
pip install -e .
```

Also requires DESHIMA-internal tools not on PyPI: `decode`, `d24_tools`,
and (for `faica_denoise.py`'s JADE rotation) the `jade` function from the
sibling [pswsc-faica-pipeline](https://github.com/ikeda46/pswsc-faica-pipeline)
repo, currently referenced via a relative `sys.path` insert rather than a
proper dependency.

## Data

This repo contains only code. Observation data (`.zarr.zip` dems files),
skydip calibration tables, and analysis outputs (plots/`.npz`/`.csv`) are
**not included** — see `.gitignore`. Results are not yet public.
