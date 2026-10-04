"""Valence/arousal estimation for one session: personal baseline -> model -> 30-s smoothing,
and the combination of that sensor reading with the tone of the user's messages."""
import numpy as np
import pandas as pd

from config import BASELINE_SEC, DISAGREE, SMOOTH_SEC, TEXT_WEIGHT
from signals import FEATURES

Z_CLIP = 5.0
TREND_DELTA = 0.25    # change of the 30-s average (1-5 scale) that counts as rising / falling


def baseline_stats(frame):
    return frame[FEATURES].mean(), frame[FEATURES].std()


def to_relative(frame, stats):
    """Features as z-scores against the person's own baseline, so the model sees
    'higher than usual for this person' rather than raw levels that differ between people."""
    mean, std = stats
    return ((frame[FEATURES] - mean) / std.replace(0, np.nan)).clip(-Z_CLIP, Z_CLIP)


class Smoother:
    """Keeps (t, valence, arousal) estimates; reports the last 30 s average and its trend."""

    def __init__(self, source):
        self.source = source
        self.history = []

    def add(self, t, valence, arousal):
        self.history.append((t, valence, arousal))

    def series(self):
        return pd.DataFrame(self.history, columns=["t", "valence", "arousal"])

    def reading(self):
        if not self.history:
            return None
        h = self.series()
        now = h.t.iloc[-1]
        recent = h[h.t > now - SMOOTH_SEC]
        before = h[(h.t <= now - SMOOTH_SEC) & (h.t > now - 2 * SMOOTH_SEC)]
        out = {"source": self.source, "t": float(now)}
        for k in ("valence", "arousal"):
            out[k] = round(float(recent[k].mean()), 2)
            change = recent[k].mean() - before[k].mean() if len(before) else 0.0
            out[f"{k}_trend"] = ("rising" if change >= TREND_DELTA else
                                 "falling" if change <= -TREND_DELTA else "steady")
        return out


class AffectEstimator:
    """Feed one window of raw features every 5 s. Windows in the first BASELINE_SEC only build the
    personal baseline; after that each window is z-scored against it and passed to the models."""

    def __init__(self, bundle, source="sensor model"):
        self.models = bundle["models"]
        self.baseline, self.stats = [], None
        self.smoother = Smoother(source)

    @property
    def calibrating(self):
        return self.stats is None

    def add(self, t, features):
        if t <= BASELINE_SEC:
            self.baseline.append(features)
            return
        if self.stats is None:
            self.stats = baseline_stats(pd.DataFrame(self.baseline, columns=FEATURES))
        x = to_relative(pd.DataFrame([features], columns=FEATURES), self.stats)
        valence, arousal = (float(np.clip(self.models[k].predict(x)[0], 1, 5)) for k in ("valence", "arousal"))
        self.smoother.add(t, valence, arousal)

    def reading(self):
        return self.smoother.reading()


def fuse(sensor, text):
    """Combined reading sent to the LLM. Sensors mostly show how activated someone is, the message
    shows whether it feels good or bad, so the message gets TEXT_WEIGHT (scaled by its confidence)
    and the sensors the rest. With only one source, that source is used as it is."""
    if text is None:
        return None if sensor is None else {**sensor, "text_share": {"valence": 0.0, "arousal": 0.0}, "disagree": []}
    out = {"source": "combined" if sensor else "message", "text_share": {}}
    for k in ("valence", "arousal"):
        w = TEXT_WEIGHT[k] * text["confidence"] if sensor else 1.0
        out[k] = round(w * text[k] + (1 - w) * (sensor[k] if sensor else 0.0), 2)
        out[f"{k}_trend"] = sensor[f"{k}_trend"] if sensor else "steady"
        out["text_share"][k] = round(w, 2)
    out["disagree"] = [k for k in ("valence", "arousal") if sensor and abs(text[k] - sensor[k]) >= DISAGREE]
    return out
