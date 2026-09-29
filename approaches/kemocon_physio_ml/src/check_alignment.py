"""Sanity check of the label-to-sensor time alignment.

Shift each person's arousal ratings by L windows (L x 5 s) relative to their sensor
features and measure the within-person correlation. If the alignment is right, the
correlation should peak at or just around lag 0; a peak far from 0 would mean the
annotation clock is offset from the sensor clock.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PROCESSED, FIGURES

FEATS = ["eda_mean", "e4_hr_mean", "temp_mean", "bvp_p2p"]
LAGS = range(-24, 25)   # +-2 minutes


def main():
    df = pd.read_csv(PROCESSED / "dataset.csv").sort_values(["pid", "window_end_s"])
    out = {f: [] for f in FEATS}
    for lag in LAGS:
        per = {f: [] for f in FEATS}
        for _, d in df.groupby("pid"):
            y = d["arousal"].shift(-lag)       # lag > 0: rating from later than the sensor window
            for f in FEATS:
                ok = y.notna() & d[f].notna()
                if ok.sum() > 30 and y[ok].nunique() > 1:
                    per[f].append(d.loc[ok, f].corr(y[ok], method="spearman"))
        for f in FEATS:
            out[f].append(np.nanmean(per[f]))
    res = pd.DataFrame(out, index=list(LAGS))
    res.to_csv(PROCESSED / "alignment_check.csv")

    fig, ax = plt.subplots(figsize=(9, 4))
    for f in FEATS:
        ax.plot(np.array(res.index) * 5, res[f], label=f, lw=1.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("label shift (s)  [positive = rating given later than the sensor window]")
    ax.set_ylabel("mean within-person Spearman r\nwith arousal")
    ax.set_title("Label/sensor alignment check")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "data_alignment_check.png", dpi=150); plt.close(fig)
    print(res.round(3).iloc[::4])


if __name__ == "__main__":
    main()
