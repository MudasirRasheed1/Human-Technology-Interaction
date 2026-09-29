"""Window-level feature extraction from raw physiological signals.

Used by the offline dataset builder and, later, by the live pipeline: both pass
raw samples (timestamps in ms + values) and a [t0, t1) window, and get back the
same feature dictionary. Only samples before t1 are ever used, so every feature
is computable in real time.

`signals` is a dict  name -> (ts, values)  with ts a sorted float array in ms:
    eda, bvp, e4_hr, ibi, temp : values shape (n,)
    acc                        : values shape (n, 3)   raw E4 units (1/64 g)
    polar_hr, attention, meditation : values shape (n,)
    eeg                        : values shape (n, 8)   order = EEG_BANDS
Any signal may be missing from the dict; its features are then NaN.
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from config import WINDOW_S, CONTEXT_S

EEG_BANDS = ["delta", "theta", "lowAlpha", "highAlpha", "lowBeta", "highBeta", "lowGamma", "middleGamma"]
EEG_COLS = ["eeg_delta", "eeg_theta", "eeg_lowAlpha", "eeg_highAlpha",
            "eeg_lowBeta", "eeg_highBeta", "eeg_lowGamma", "eeg_midGamma"]

EDA_HZ = 4
SCR_MIN_AMP = 0.01      # uS; standard minimum amplitude for a skin-conductance response
IBI_RANGE = (300, 2000)  # ms; 30-200 bpm, anything outside is an artefact
NAN = float("nan")


def _slice(signals, name, t0, t1):
    if name not in signals:
        return np.empty(0), np.empty(0)
    ts, v = signals[name]
    i0, i1 = np.searchsorted(ts, t0, "left"), np.searchsorted(ts, t1, "left")
    return ts[i0:i1], v[i0:i1]


def _clean(ts, v):
    ok = np.isfinite(v)
    return ts[ok], v[ok]


def _slope(ts, v):
    """Least-squares slope in units per second."""
    if len(v) < 3 or np.ptp(ts) == 0:
        return NAN
    return float(np.polyfit((ts - ts[0]) / 1000.0, v, 1)[0])


def _basic(ts, v, prefix, stats=("mean", "std", "slope")):
    ts, v = _clean(ts, v)
    out = {}
    for s in stats:
        key = f"{prefix}_{s}"
        if len(v) == 0:
            out[key] = NAN
        elif s == "mean":
            out[key] = float(v.mean())
        elif s == "std":
            out[key] = float(v.std()) if len(v) > 1 else NAN
        elif s == "min":
            out[key] = float(v.min())
        elif s == "max":
            out[key] = float(v.max())
        elif s == "slope":
            out[key] = _slope(ts, v)
    return out


def _scr_peaks(ts, v):
    """Skin-conductance responses: peaks of the 1 s-smoothed EDA with prominence >= SCR_MIN_AMP."""
    ts, v = _clean(ts, v)
    if len(v) < 2 * EDA_HZ:
        return None, None
    padded = np.pad(v, (EDA_HZ // 2, EDA_HZ - 1 - EDA_HZ // 2), mode="edge")  # no zero-padding dips at the edges
    smooth = np.convolve(padded, np.ones(EDA_HZ) / EDA_HZ, mode="valid")
    idx, props = find_peaks(smooth, prominence=SCR_MIN_AMP)
    return ts[idx], props["prominences"]


def _ibi_hrv(ts, v, prefix):
    ts, v = _clean(ts, v)
    keep = (v >= IBI_RANGE[0]) & (v <= IBI_RANGE[1])
    ts, v = ts[keep], v[keep]
    out = {}
    if prefix == "ibi":
        out["ibi_mean"] = float(v.mean()) if len(v) else NAN
        out["ibi_count"] = float(len(v))
    out[f"{prefix}_sdnn"] = float(v.std()) if len(v) >= 2 else NAN
    # RMSSD only over truly successive beats: the E4 drops beats it cannot detect,
    # so two consecutive rows are neighbours only if the time gap matches the interval.
    if len(v) >= 2:
        successive = np.abs(np.diff(ts) - v[1:]) < 100
        d = np.diff(v)[successive]
        out[f"{prefix}_rmssd"] = float(np.sqrt(np.mean(d ** 2))) if len(d) >= 1 else NAN
    else:
        out[f"{prefix}_rmssd"] = NAN
    return out


def window_features(signals, t0, t1):
    """All features for the window [t0, t1) (ms). Trailing context uses [t1 - CONTEXT_S, t1)."""
    c0 = t1 - CONTEXT_S * 1000
    f = {}

    # --- EDA (A, B, context) ---
    ts, v = _slice(signals, "eda", t0, t1)
    f.update(_basic(ts, v, "eda", ("mean", "std", "min", "max", "slope")))
    f["q_eda_frac"] = min(1.0, np.isfinite(v).sum() / (WINDOW_S * EDA_HZ))
    cts, cv = _slice(signals, "eda", c0, t1)
    ptimes, pamps = _scr_peaks(cts, cv)
    if ptimes is None:
        f["eda_scr_count"] = f["eda_scr_amp"] = f["eda_scr_count_30s"] = NAN
    else:
        inwin = ptimes >= t0
        f["eda_scr_count"] = float(inwin.sum())
        f["eda_scr_amp"] = float(pamps[inwin].mean()) if inwin.any() else 0.0
        f["eda_scr_count_30s"] = float(len(ptimes))
    ctx = _basic(cts, cv, "eda", ("mean", "slope"))
    f["eda_mean_30s"], f["eda_slope_30s"] = ctx["eda_mean"], ctx["eda_slope"]

    # --- BVP (C): robust peak-to-peak amplitude of the pulse wave ---
    _, v = _clean(*_slice(signals, "bvp", t0, t1))
    f["bvp_p2p"] = float(np.percentile(v, 95) - np.percentile(v, 5)) if len(v) >= 64 else NAN

    # --- E4 heart rate (D, context) ---
    f.update(_basic(*_slice(signals, "e4_hr", t0, t1), "e4_hr"))
    f["e4_hr_mean_30s"] = _basic(*_slice(signals, "e4_hr", c0, t1), "x", ("mean",))["x_mean"]

    # --- IBI / HRV (E, context) ---
    f.update(_ibi_hrv(*_slice(signals, "ibi", t0, t1), "ibi"))
    f.update(_ibi_hrv(*_slice(signals, "ibi", c0, t1), "ibi_30s"))
    f["ibi_sdnn_30s"], f["ibi_rmssd_30s"] = f.pop("ibi_30s_sdnn"), f.pop("ibi_30s_rmssd")

    # --- Skin temperature (F) ---
    f.update(_basic(*_slice(signals, "temp", t0, t1), "temp", ("mean", "slope")))

    # --- Accelerometer (G): magnitude in g ---
    _, a = _slice(signals, "acc", t0, t1)
    if len(a):
        mag = np.linalg.norm(a.astype(float), axis=1) / 64.0
        f["acc_mag_mean"], f["acc_mag_std"] = float(mag.mean()), float(mag.std())
    else:
        f["acc_mag_mean"] = f["acc_mag_std"] = NAN

    # --- Polar heart rate (H) ---
    f.update(_basic(*_slice(signals, "polar_hr", t0, t1), "polar_hr"))

    # --- EEG band powers (I) and ratios (J) ---
    _, e = _slice(signals, "eeg", t0, t1)
    if len(e):
        e = e.astype(float)
        logp = np.log10(e + 1)
        for col, m in zip(EEG_COLS, logp.mean(axis=0)):
            f[col] = float(m)
        alpha, beta, theta = e[:, 2] + e[:, 3], e[:, 4] + e[:, 5], e[:, 1]
        f["eeg_alpha_beta_ratio"] = float(np.mean(np.log10((alpha + 1) / (beta + 1))))
        f["eeg_theta_beta_ratio"] = float(np.mean(np.log10((theta + 1) / (beta + 1))))
        f["q_eeg_n"] = float(len(e))
    else:
        for col in EEG_COLS + ["eeg_alpha_beta_ratio", "eeg_theta_beta_ratio"]:
            f[col] = NAN
        f["q_eeg_n"] = 0.0

    # --- NeuroSky eSense (K) ---
    f["attention_mean"] = _basic(*_slice(signals, "attention", t0, t1), "x", ("mean",))["x_mean"]
    f["meditation_mean"] = _basic(*_slice(signals, "meditation", t0, t1), "x", ("mean",))["x_mean"]
    return f


# ---------------------------------------------------------------------------
# Per-person normalisation (applied at training time and in the live pipeline)
# ---------------------------------------------------------------------------

def stats_from_baseline(baseline_stats, features):
    """Per-person (mean, std) frames from baseline_stats.csv (indexed by pid)."""
    mu = baseline_stats[[f"{f}__mean" for f in features]].set_axis(features, axis=1)
    sd = baseline_stats[[f"{f}__std" for f in features]].set_axis(features, axis=1)
    return mu, sd


def stats_from_calibration(df, features, n_windows):
    """Per-person (mean, std) over each person's first `n_windows` conversation windows (no labels needed)."""
    first = df.sort_values(["pid", "window_end_s"]).groupby("pid").head(n_windows)
    g = first.groupby("pid")[features]
    return g.mean(), g.std()


def normalise(df, features, mu, sd, floor):
    """z = (x - person_mean) / (person_std + floor).

    `floor` (per feature, = median person std in the training people) stops a near-constant
    reference period from blowing values up.
    """
    m = mu.reindex(df["pid"])[features].to_numpy()
    s = sd.reindex(df["pid"])[features].fillna(0).to_numpy()
    X = (df[features].to_numpy() - m) / (s + floor[features].to_numpy())
    return pd.DataFrame(X, columns=features, index=df.index)


def session_normalise(df, features):
    """z-score within each person's own debate windows (reference only: needs the whole session)."""
    out = df.copy()
    g = df.groupby("pid")[features]
    out[features] = (df[features] - g.transform("mean")) / (g.transform("std") + 1e-9)
    return out
