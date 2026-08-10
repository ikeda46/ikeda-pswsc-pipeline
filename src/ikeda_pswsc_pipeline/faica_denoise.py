"""Step 1-4: FA + JADE (no whitening) denoising, per beam.

Procedure (per beam, on that beam's full ON+OFF signal, GRAD excluded):
1. Subtract the per-channel mean (kept aside for later reconstruction).
2. Factor Analysis; number of components chosen by AIC.
3. JADE rotation applied directly to FA's factor scores -- no separate
   whitening step (unlike `pswsc_faica_pipeline.faica`'s `whiten_fa`+`jade`
   combination).
4. Identify "strange" ICs and clean/remove them (denoising) -- criteria
   below. NOT YET IMPLEMENTED as an automatic filter; this module currently
   only computes the diagnostics needed to flag candidates for review.
5. Reconstruct from the kept/cleaned ICs, add the mean back.

Strange-IC criteria (2026-08-10 discussion):
1. |corr| with the on-off square-wave reference < 0.5 -> removal candidate.
2. A level jump partway through, or signal confined to a short duration ->
   don't discard the whole IC (would bias the overall mean) -- just clean
   the jump/outlier samples out of its time series.
3. Persistent power at relatively high frequency (i.e. noise-like) ->
   discard the whole IC.
4. Loading concentrated on very few channels -> discard the whole IC,
   UNLESS already covered by criterion 2 or 3 (then no extra action).
"""

from __future__ import annotations

import sys

import numpy as np
from sklearn.decomposition import FactorAnalysis

sys.path.insert(0, "../../pswsc-faica-pipeline/src")
from pswsc_faica_pipeline.faica import jade  # noqa: E402

ON_BEAM_FOR_STATE = {"ON": "A", "OFF": "B"}


def _free_params(n_features: int, m: int) -> int:
    """FA free-parameter count: loadings (n_features*m, corrected for the
    m*(m-1)/2-dimensional rotational indeterminacy) + n_features diagonal
    noise variances. Ledermann bound (max identifiable m) follows from
    requiring this to not exceed n_features*(n_features+1)/2:
    m_max = (2*n_features+1 - sqrt(8*n_features+1)) / 2."""
    return n_features * m - m * (m - 1) // 2 + n_features


def select_n_components_mdl(X: np.ndarray, max_components: int = 15, verbose: bool = True) -> tuple[int, np.ndarray]:
    """Pick FactorAnalysis's n_components by MDL (Rissanen):
    MDL(m) = -logL(m) + 0.5*k(m)*log(N) -- same free-parameter count k(m)
    as AIC, just penalized by 0.5*log(N) per parameter instead of a flat 2
    (heavier penalty for the typically-large N here, so MDL selects fewer
    components than AIC would). X should already be centered.

    verbose: print each m's logL/MDL as it's computed (flushed immediately),
    so a long search can be monitored live instead of only reporting at
    the end (each FA fit here can take a while for large m/n_features).
    """
    import sys
    import time

    n_samples, n_features = X.shape
    mdls = np.full(max_components, np.nan)
    for i, m in enumerate(range(1, max_components + 1)):
        t0 = time.time()
        fa = FactorAnalysis(n_components=m, random_state=0)
        fa.fit(X)
        logL = fa.score(X) * n_samples
        k = _free_params(n_features, m)
        mdls[i] = -logL + 0.5 * k * np.log(n_samples)
        if verbose:
            print(f"    m={m:3d}  logL={logL:14.1f}  k={k:6d}  MDL={mdls[i]:14.1f}  "
                  f"({time.time() - t0:.1f}s)", flush=True)
            sys.stdout.flush()
    best_m = int(np.nanargmin(mdls)) + 1
    return best_m, mdls


def fit_faica(X: np.ndarray, max_components: int = 15, n_components: int | None = None) -> dict:
    """Steps 1-3: center, FA (MDL-selected OR fixed), no-whitening JADE.

    X: (T, n_channels), NOT yet centered.

    n_components: if given, use this m directly and SKIP the MDL search
    entirely (`mdls` in the result will be None). Useful when MDL doesn't
    turn around within a computationally reasonable range and a fixed m
    is used instead (2026-08-10: MDL kept decreasing all the way to the
    JADE-cost ceiling around m=60 for a 250-channel/~70000-sample pswsc
    beam, so model-selection was dropped in favor of a fixed m=20).

    Returns S (m, T) independent components, per-channel loadings
    A_mix (n_channels, m), the channel means, and the MDL curve (or None).
    """
    mean = np.nanmean(X, axis=0)
    Xc = X - mean[None, :]

    if n_components is not None:
        m, mdls = n_components, None
    else:
        m, mdls = select_n_components_mdl(Xc, max_components)
    fa = FactorAnalysis(n_components=m, random_state=0)
    fa.fit(Xc)
    F = fa.transform(Xc)  # (T, m) factor scores -- NOT whitened via whiten_fa
    Z = F.T  # (m, T)

    V = jade(Z)
    S = V @ Z  # (m, T) independent components

    Lambda = fa.components_.T  # (n_channels, m)
    A_mix = Lambda @ V.T  # (n_channels, m) -- per-IC per-channel loading

    return dict(S=S, A_mix=A_mix, mean=mean, m=m, mdls=mdls, fa=fa, V=V)


def clean_ic_spike(s: np.ndarray, top_frac: float = 0.05) -> np.ndarray:
    """Criterion-2 cleanup: replace the most extreme `top_frac` of an IC's
    samples (by squared value -- the same definition `frac_energy_top5pct`
    screens on) with the IC's OWN median, instead of discarding the whole
    IC (which would bias the reconstruction, per the 2026-08-10 discussion).
    """
    T = s.size
    n_cut = max(1, int(round(top_frac * T)))
    cut_idx = np.argsort(s ** 2)[::-1][:n_cut]
    s_clean = s.copy()
    s_clean[cut_idx] = np.median(s)
    return s_clean


def ic_diagnostics(S: np.ndarray, A_mix: np.ndarray, onoff_ref: np.ndarray,
                    fs: float, n_jump_segments: int = 10,
                    high_freq_hz: float = 0.2) -> list[dict]:
    """Per-IC diagnostics for the 4 strange-IC criteria (2026-08-10)."""
    m, T = S.shape
    diags = []
    for k in range(m):
        s = S[k]

        # criterion 1: |corr| with on-off reference
        corr = np.corrcoef(s, onoff_ref)[0, 1]

        # criterion 2: jump / short-duration-signal score -- split into
        # segments, look for a large jump between adjacent segment means
        # relative to the overall sample std, and check how much of the
        # total squared deviation is concentrated in a small time fraction.
        seg_edges = np.linspace(0, T, n_jump_segments + 1).astype(int)
        seg_means = np.array([np.mean(s[seg_edges[i]:seg_edges[i + 1]]) for i in range(n_jump_segments)])
        jump_score = float(np.max(np.abs(np.diff(seg_means))) / (np.std(s) + 1e-30))
        frac_energy_top5pct = float(np.sum(np.sort(s ** 2)[::-1][:max(1, T // 20)]) / np.sum(s ** 2))

        # criterion 3: fraction of power above high_freq_hz (Welch PSD)
        from scipy.signal import welch
        freqs, psd = welch(s, fs=fs, nperseg=min(4096, T))
        high_frac = float(np.sum(psd[freqs >= high_freq_hz]) / np.sum(psd))

        # criterion 4: loading concentration (inverse participation ratio,
        # normalized to [0,1]; 1/n_chan = spread evenly, 1 = one channel only)
        w = A_mix[:, k]
        w2 = w ** 2
        ipr = float(np.sum(w2 ** 2) / (np.sum(w2) ** 2 + 1e-300))

        diags.append(dict(k=k, corr=corr, abs_corr=abs(corr), jump_score=jump_score,
                          frac_energy_top5pct=frac_energy_top5pct, high_freq_frac=high_frac,
                          loading_ipr=ipr, n_chan_eff=1.0 / ipr if ipr > 0 else np.nan))
    return diags


DEFAULT_NOISY_KID_THRESH = 10.0


def flag_noisy_kids(noise_variance_A: np.ndarray, noise_variance_B: np.ndarray,
                     thresh_factor: float = DEFAULT_NOISY_KID_THRESH) -> np.ndarray:
    """ADOPTED (2026-08-10) noisy-KID flag: simpler and more robust than
    IC-removal-and-reconstruct (which was found to have a side effect --
    dropping a whole criterion-4 IC removes not just the offending
    channel's dominant loading but also that IC's smaller, genuinely
    useful loadings on OTHER channels, which had been helping cancel
    correlated noise there -- so the "denoised" spectrum ended up NOISIER
    in several bands than the plain LPF-only spectrum).

    Per-KID noise std (sqrt(fa.noise_variance_), from a fixed-m=20 FA fit
    on that beam's full ON+OFF signal) is compared against `thresh_factor`
    times the median std -- a KID is flagged only if it exceeds this in
    BOTH beams independently (agreement between two independently-fit
    beams is a more robust criterion than either alone). Flagged KIDs are
    meant to be marked on the plain (non-FA-denoised) spectrum, not used
    to reconstruct a cleaned time series.

    thresh_factor=10.0 was chosen empirically (2026-08-10, NGC1068
    obsid=20240715094140): 5x flagged 6/250 KIDs including two borderline
    ones; 10x flagged only the 2 KIDs (265.5, 321.5 GHz) that are also
    the two visually obvious outliers in the spectrum's own error bars.
    """
    std_A = np.sqrt(noise_variance_A)
    std_B = np.sqrt(noise_variance_B)
    return (std_A > thresh_factor * np.median(std_A)) & (std_B > thresh_factor * np.median(std_B))
