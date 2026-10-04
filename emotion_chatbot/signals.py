"""
Raw E4 wristband signals -> one feature vector per 10-second window.

The same window_features() is used for training (K-EmoCon recordings), for the app's replay and
later for a live device, so the model always sees features computed the same way.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from config import KEMOCON_ROOT, WINDOW_SEC

FS = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}   # E4 sampling rates (HR is 1 Hz)
STREAMS = ["BVP", "EDA", "HR", "TEMP", "ACC"]
FEATURES = ["eda_mean", "eda_std", "eda_slope", "eda_range", "eda_scr",
            "bvp_hr", "bvp_rmssd", "bvp_amp", "hr_mean", "hr_slope",
            "temp_mean", "temp_slope", "movement"]
_BVP_BAND = signal.butter(2, [0.5, 8], btype="band", fs=FS["BVP"], output="sos")


def _slope_per_s(x, fs):
    return float(np.polyfit(np.arange(len(x)) / fs, x, 1)[0]) if len(x) >= 2 else np.nan


def window_features(w):
    """w: stream name -> samples inside one window (ACC is an (n, 3) array).
    A missing or too-short stream leaves its features as NaN; the model handles NaN."""
    f = dict.fromkeys(FEATURES, np.nan)

    eda = np.asarray(w.get("EDA", []), float)
    eda = eda[eda > 0]                                    # 0 uS means the electrodes lost contact
    if len(eda) >= WINDOW_SEC * FS["EDA"] // 2:
        f.update(eda_mean=eda.mean(), eda_std=eda.std(), eda_slope=_slope_per_s(eda, FS["EDA"]),
                 eda_range=np.ptp(eda), eda_scr=len(signal.find_peaks(eda, prominence=0.01)[0]))

    bvp = np.asarray(w.get("BVP", []), float)
    if len(bvp) >= WINDOW_SEC * FS["BVP"] // 2:
        bvp = signal.sosfiltfilt(_BVP_BAND, bvp)
        peaks, _ = signal.find_peaks(bvp, distance=int(0.33 * FS["BVP"]), prominence=0.5 * bvp.std())
        ibi = np.diff(peaks) / FS["BVP"] * 1000
        ibi = ibi[(ibi > 300) & (ibi < 2000)]            # 30-200 bpm
        f["bvp_amp"] = bvp.std()
        if len(ibi) >= 3:
            f.update(bvp_hr=60000 / ibi.mean(), bvp_rmssd=np.sqrt(np.mean(np.diff(ibi) ** 2)))

    hr = np.asarray(w.get("HR", []), float)
    hr = hr[hr > 0]
    if len(hr) >= 2:
        f.update(hr_mean=hr.mean(), hr_slope=_slope_per_s(hr, 1))

    temp = np.asarray(w.get("TEMP", []), float)
    if len(temp) >= 2:
        f.update(temp_mean=temp.mean(), temp_slope=_slope_per_s(temp, FS["TEMP"]))

    acc = np.asarray(w.get("ACC", np.empty((0, 3))), float).reshape(-1, 3)
    if len(acc) >= 2:
        f["movement"] = np.linalg.norm(acc, axis=1).std()
    return f


def kemocon_participants(root=KEMOCON_ROOT):
    """Participants with all E4 streams the model needs and self-annotations."""
    avail = pd.read_csv(Path(root) / "metadata" / "metadata" / "data_availability.csv").set_index("pid")
    need = ["E4_BVP", "E4_EDA", "E4_HR", "E4_TEMP", "self_annotations"]
    return [int(p) for p in avail.index if avail.loc[p, need].astype(bool).all()]


class Recording:
    """One K-EmoCon participant's E4 streams, sliced by seconds since their debate started."""

    def __init__(self, pid, root=KEMOCON_ROOT):
        self.pid, self.root = pid, Path(root)
        folder = self.root / "e4_data" / "e4_data" / str(pid)
        self.streams = {}
        for name in STREAMS:
            fp = folder / f"E4_{name}.csv"
            if fp.exists():
                df = pd.read_csv(fp).sort_values("timestamp").drop_duplicates("timestamp")
                cols = ["x", "y", "z"] if name == "ACC" else "value"
                self.streams[name] = (df["timestamp"].to_numpy(float), df[cols].to_numpy(float))
        subj = pd.read_csv(self.root / "metadata" / "metadata" / "subjects.csv").set_index("pid").loc[pid]
        self.start_ms, self.end_ms = float(subj["startTime"]), float(subj["endTime"])

    @property
    def duration_s(self):
        return (self.end_ms - self.start_ms) / 1000

    def window(self, end_s, length_s=WINDOW_SEC):
        """Samples in [end_s - length_s, end_s)."""
        t0, t1 = self.start_ms + (end_s - length_s) * 1000, self.start_ms + end_s * 1000
        out = {}
        for name, (ts, values) in self.streams.items():
            i, j = np.searchsorted(ts, [t0, t1])
            out[name] = values[i:j]
        return out

    def features(self, end_s):
        return window_features(self.window(end_s))

    def self_report(self):
        """The participant's own 1-5 valence/arousal ratings, one every 5 s."""
        fp = (self.root / "emotion_annotations" / "emotion_annotations" / "self_annotations"
              / f"P{self.pid}.self.csv")
        df = pd.read_csv(fp, na_values=["x", "X", ""])
        df.columns = [c.strip() for c in df.columns]
        return df[["seconds", "arousal", "valence"]].apply(pd.to_numeric, errors="coerce").dropna()
