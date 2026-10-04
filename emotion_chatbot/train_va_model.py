"""
Trains the valence and arousal models used by the chatbot, from raw K-EmoCon E4 recordings.

  1. For every 5-s self-annotation (1-5 valence/arousal), features of the preceding 10 s of
     wristband signal (signals.py).
  2. Person-relative features: z-scores against that person's first 2 minutes, exactly as the app
     does for a live user. Those 2 minutes are not used as training targets.
  3. Leave-one-participant-out evaluation, so the reported quality is for an unseen person.
  4. Refit on everyone and save models/va_model.joblib (with the evaluation, shown in the app).

Run from this folder with the `torch` conda env:
    python train_va_model.py
"""
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from affect_model import baseline_stats, to_relative
from config import BASELINE_SEC, KEMOCON_ROOT, MODEL_PATH, STEP_SEC, WINDOW_SEC
from signals import FEATURES, Recording, kemocon_participants

TARGETS = ["valence", "arousal"]


def make_model():
    # Heavily regularised: on unseen people, larger models and ridge regression were all further
    # from the true ratings than this one.
    return HistGradientBoostingRegressor(max_iter=60, learning_rate=0.03, max_depth=2,
                                         min_samples_leaf=150, l2_regularization=10.0, random_state=0)


def build_dataset():
    frames = []
    for pid in kemocon_participants():
        rec = Recording(pid)
        labels = rec.self_report()
        rows = [dict(pid=pid, seconds=s, arousal=a, valence=v, **rec.features(s))
                for s, a, v in labels[["seconds", "arousal", "valence"]].itertuples(index=False)]
        df = pd.DataFrame(rows)
        stats = baseline_stats(df[df.seconds <= BASELINE_SEC])
        df[FEATURES] = to_relative(df, stats)
        frames.append(df[df.seconds > BASELINE_SEC])
        print(f"  P{pid}: {len(df)} windows")
    return pd.concat(frames, ignore_index=True)


def evaluate(df):
    """Leave one participant out: train on everyone else, predict the held-out person."""
    pred = {t: np.full(len(df), np.nan) for t in TARGETS}
    base = {t: np.full(len(df), np.nan) for t in TARGETS}
    for pid in df.pid.unique():
        test = (df.pid == pid).to_numpy()
        for t in TARGETS:
            model = make_model().fit(df.loc[~test, FEATURES], df.loc[~test, t])
            pred[t][test] = np.clip(model.predict(df.loc[test, FEATURES]), 1, 5)
            base[t][test] = df.loc[~test, t].mean()
    metrics = {}
    for t in TARGETS:
        y = df[t].to_numpy()
        within = []
        for pid in df.pid.unique():
            m = (df.pid == pid).to_numpy()
            if np.std(y[m]) > 0 and np.std(pred[t][m]) > 0:
                within.append(spearmanr(y[m], pred[t][m]).statistic)
        metrics[t] = {"MAE": float(np.mean(np.abs(pred[t] - y))),
                      "MAE (always predict the average)": float(np.mean(np.abs(base[t] - y))),
                      "within-person correlation (median)": float(np.median(within)),
                      "people with positive correlation": f"{int(np.sum(np.array(within) > 0))}/{len(within)}"}
    return metrics


def main():
    t0 = time.time()
    print(f"Reading K-EmoCon from {KEMOCON_ROOT}")
    df = build_dataset()
    print(f"{len(df)} windows from {df.pid.nunique()} participants (after each person's 2-min baseline)\n")

    metrics = evaluate(df)
    print("Leave-one-participant-out (scale 1-5):")
    print(pd.DataFrame(metrics).round(3).to_string(), "\n")

    bundle = {"features": FEATURES,
              "models": {t: make_model().fit(df[FEATURES], df[t]) for t in TARGETS},
              "metrics": metrics,
              "label_mean": {t: float(df[t].mean()) for t in TARGETS},
              "trained_on": f"K-EmoCon self-annotations, {df.pid.nunique()} participants, {len(df)} windows",
              "settings": {"window_sec": WINDOW_SEC, "step_sec": STEP_SEC, "baseline_sec": BASELINE_SEC}}
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved {MODEL_PATH} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
