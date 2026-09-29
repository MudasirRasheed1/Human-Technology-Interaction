"""Build the windowed K-EmoCon dataset.

One row = one 5-second window of one participant's debate:
    identifiers | ~40 sensor features | self-reported valence/arousal (+ 3-class versions) | aux emotions

Outputs (data/processed/):
    dataset.csv          all usable windows
    train.csv, test.csv  split by debate pair (no person or debate partner in both)
    baseline_stats.csv   per-person mean/std of every feature over the 3 min before the debate
    split.json, dataset_summary.json
"""
import json

import numpy as np
import pandas as pd

from config import (RAW, PROCESSED, WINDOW_S, BASELINE_S, MIN_EDA_COVERAGE, RANDOM_SEED,
                    N_TEST_PAIRS, ALL_FEATURES, to_class)
from features import window_features, EEG_BANDS

RATED_EMOTIONS = ["cheerful", "happy", "angry", "nervous", "sad"]          # 1-4 scale
MARKED_EMOTIONS = ["boredom", "confusion", "delight", "concentration", "frustration", "surprise",
                   "confrustion", "contempt", "dejection", "disgust", "eureka", "pride", "sorrow"]  # 'x' = present


def _read(path, cols):
    df = pd.read_csv(path, usecols=["timestamp"] + cols).sort_values("timestamp")
    return df["timestamp"].to_numpy(float), df[cols].to_numpy(float).squeeze()


def load_signals(pid):
    e4, nsp = RAW / "e4_data" / str(pid), RAW / "neurosky_polar_data" / str(pid)
    sig = {}
    for name, fname in [("eda", "E4_EDA"), ("bvp", "E4_BVP"), ("e4_hr", "E4_HR"), ("ibi", "E4_IBI"), ("temp", "E4_TEMP")]:
        p = e4 / f"{fname}.csv"
        if p.exists():
            sig[name] = _read(p, ["value"])
    if (e4 / "E4_ACC.csv").exists():
        sig["acc"] = _read(e4 / "E4_ACC.csv", ["x", "y", "z"])
    for name, fname in [("polar_hr", "Polar_HR"), ("attention", "Attention"), ("meditation", "Meditation")]:
        p = nsp / f"{fname}.csv"
        if p.exists():
            sig[name] = _read(p, ["value"])
    if (nsp / "BrainWave.csv").exists():
        sig["eeg"] = _read(nsp / "BrainWave.csv", EEG_BANDS)

    # A reading of exactly 0 means "no contact / no value" for these devices, not a real measurement.
    for name in ["eda", "e4_hr", "temp", "polar_hr", "attention", "meditation"]:
        if name in sig:
            ts, v = sig[name]
            sig[name] = (ts, np.where(v == 0, np.nan, v))
    return sig


def load_labels(pid):
    a = pd.read_csv(RAW / "emotion_annotations" / "self_annotations" / f"P{pid}.self.csv")
    out = a[["seconds", "arousal", "valence"]].copy()
    for c in RATED_EMOTIONS:
        out[f"aux_{c}"] = pd.to_numeric(a[c], errors="coerce")
    for c in MARKED_EMOTIONS:
        out[f"aux_{c}"] = (a[c].astype(str).str.strip().str.lower() == "x").astype(int)
    return out


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(RAW / "metadata" / "subjects.csv").set_index("pid")
    avail = pd.read_csv(RAW / "metadata" / "data_availability.csv").set_index("pid")

    rows, base_rows, log = [], [], []
    for pid, m in meta.iterrows():
        labels = load_labels(pid)
        entry = {"pid": int(pid), "labelled_windows": len(labels)}
        if not avail.loc[pid, "E4_EDA"]:
            entry.update(kept=0, dropped_no_e4=len(labels))
            log.append(entry)
            print(f"P{pid:2d}: no E4 data -> excluded")
            continue

        sig = load_signals(pid)
        kept = beyond_end = no_eda = 0
        for _, lab in labels.iterrows():
            t1 = m.startTime + lab.seconds * 1000
            t0 = t1 - WINDOW_S * 1000
            if t0 >= m.endTime:          # annotation runs past the end of the sensor recording
                beyond_end += 1
                continue
            f = window_features(sig, t0, t1)
            if f["q_eda_frac"] < MIN_EDA_COVERAGE:
                no_eda += 1
                continue
            row = {"pid": int(pid), "pair": (int(pid) + 1) // 2,
                   "window_start_s": int(lab.seconds - WINDOW_S), "window_end_s": int(lab.seconds)}
            row.update(f)
            row.update({"valence": int(lab.valence), "arousal": int(lab.arousal),
                        "valence_cls": to_class(lab.valence), "arousal_cls": to_class(lab.arousal)})
            row.update({k: lab[k] for k in labels.columns if k.startswith("aux_")})
            rows.append(row)
            kept += 1

        # Baseline: 5 s windows over the last BASELINE_S seconds before the debate starts.
        bl = [window_features(sig, t1 - WINDOW_S * 1000, t1)
              for t1 in np.arange(m.startTime - BASELINE_S * 1000 + WINDOW_S * 1000, m.startTime + 1, WINDOW_S * 1000)]
        bl = pd.DataFrame(bl)[ALL_FEATURES]
        stats = {"pid": int(pid)}
        for c in ALL_FEATURES:
            stats[f"{c}__mean"], stats[f"{c}__std"] = bl[c].mean(), bl[c].std()
        base_rows.append(stats)

        entry.update(kept=kept, dropped_beyond_end=beyond_end, dropped_no_eda=no_eda)
        log.append(entry)
        print(f"P{pid:2d}: kept {kept:3d} / {len(labels):3d}  (past end {beyond_end}, no EDA {no_eda})")

    df = pd.DataFrame(rows)
    id_cols = ["pid", "pair", "window_start_s", "window_end_s"]
    label_cols = ["valence", "arousal", "valence_cls", "arousal_cls"]
    q_cols = ["q_eda_frac", "q_eeg_n"]
    aux_cols = [c for c in df.columns if c.startswith("aux_")]
    df = df[id_cols + ALL_FEATURES + q_cols + label_cols + aux_cols]

    # Keep only people with a meaningful amount of data (>= 30 windows = 2.5 min).
    counts = df.pid.value_counts()
    too_few = sorted(counts[counts < 30].index.tolist())
    df = df[~df.pid.isin(too_few)].reset_index(drop=True)
    for e in log:
        if e["pid"] in too_few:
            e["excluded_too_few_windows"] = True

    # Train/test split by debate pair: partners share a debate, topic and timeline,
    # so both go to the same side. Only pairs with both members present are test candidates.
    pair_sizes = df.groupby("pair").pid.nunique()
    full_pairs = sorted(pair_sizes[pair_sizes == 2].index.tolist())
    rng = np.random.default_rng(RANDOM_SEED)
    test_pairs = sorted(rng.choice(full_pairs, N_TEST_PAIRS, replace=False).tolist())
    test_mask = df.pair.isin(test_pairs)
    train, test = df[~test_mask], df[test_mask]

    df.to_csv(PROCESSED / "dataset.csv", index=False)
    train.to_csv(PROCESSED / "train.csv", index=False)
    test.to_csv(PROCESSED / "test.csv", index=False)
    base = pd.DataFrame(base_rows)
    base[base.pid.isin(df.pid.unique())].to_csv(PROCESSED / "baseline_stats.csv", index=False)

    split = {"test_pairs": test_pairs, "test_pids": sorted(test.pid.unique().tolist()),
             "train_pids": sorted(train.pid.unique().tolist()), "seed": RANDOM_SEED}
    json.dump(split, open(PROCESSED / "split.json", "w"), indent=2)

    summary = {
        "rows": len(df), "columns": df.shape[1], "features": len(ALL_FEATURES),
        "participants": int(df.pid.nunique()), "train_rows": len(train), "test_rows": len(test),
        "excluded_too_few_windows": too_few, "per_participant": log,
        "missing_fraction": df[ALL_FEATURES].isna().mean().round(4).to_dict(),
    }
    json.dump(summary, open(PROCESSED / "dataset_summary.json", "w"), indent=2, default=float)

    print(f"\ndataset: {df.shape}  participants={df.pid.nunique()}  excluded(too few)={too_few}")
    print(f"train: {train.shape} pids={split['train_pids']}")
    print(f"test : {test.shape} pids={split['test_pids']}")
    for t in ["valence_cls", "arousal_cls"]:
        print(t, "train", train[t].value_counts(normalize=True).sort_index().round(3).to_dict(),
              "test", test[t].value_counts(normalize=True).sort_index().round(3).to_dict())


if __name__ == "__main__":
    main()
