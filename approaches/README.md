# Approaches

One folder per approach. Copy `_template/` to start a new one, then add a row here.

| Folder | Owner | Dataset | Inputs | Method | Status | Headline result (on unseen people) |
|---|---|---|---|---|---|---|
| [`kemocon_physio_ml`](kemocon_physio_ml/) | Mudasir | K-EmoCon | Empatica E4 + NeuroSky EEG | Random Forest on 5 s window features + 3-min per-person calibration, abstains when unsure | Done (step 1) | **Arousal:** 60% accuracy on 3 classes (69% on the 28% of windows it answers confidently) vs 33% majority class. **Valence:** not predictable from body sensors (37% vs 51% majority) |
| | | | | | | |
