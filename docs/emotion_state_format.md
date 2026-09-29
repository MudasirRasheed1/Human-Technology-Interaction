# Common emotion-state output format

Every approach (sensors, voice, text, face, …) should be able to emit its estimate in this format. Estimators can then be swapped, compared and fused before the result is summarised for the LLM.

```json
{
  "timestamp": "2026-09-30T14:05:10Z",
  "window_s": 5,
  "source": "kemocon_physio_ml",
  "arousal": {
    "level": "high",
    "score": 4.1,
    "confidence": 0.78,
    "status": "confident"
  },
  "valence": {
    "level": null,
    "score": null,
    "confidence": 0.41,
    "status": "unknown"
  },
  "notes": "signal quality good; 3 skin-conductance responses in last 30 s"
}
```

| Field | Meaning |
|---|---|
| `level` | `"low"`, `"neutral"` or `"high"`; `null` when unknown |
| `score` | 1–5 scale estimate (optional); `null` when unknown |
| `confidence` | 0–1, the estimator's own confidence in `level` |
| `status` | `"confident"` if `confidence` ≥ the approach's validated threshold, otherwise `"unknown"` |
| `notes` | optional human-readable context (signal quality, events) |

**Rule:** the LLM only ever receives `status: "confident"` values, or descriptions derived from them. It never receives raw sensor data.
