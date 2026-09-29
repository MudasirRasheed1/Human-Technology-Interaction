# Literature, datasets and pretrained models

Collected in September 2026 while working on `approaches/kemocon_physio_ml`.

## Datasets with body sensors and valence/arousal labels

| Dataset | People | Sensors | Labels | Notes |
|---|---|---|---|---|
| **K-EmoCon** | 32 | Empatica E4, Polar H7, NeuroSky EEG, audio/video | self, partner and observer ratings (1–5) every 5 s | **Only dataset with natural conversations** (debates). Request access via Zenodo |
| **CEAP-360VR** | 32 | Empatica E4, eye tracking | continuous joystick valence/arousal | Same wristband as K-EmoCon; 360° VR videos. CC BY-NC. [site](https://www.dis.cwi.nl/ceap-360vr-dataset/), [GitHub](https://github.com/cwi-dis/CEAP-360VR-Dataset) |
| **CASE** | 30 | ECG, BVP, EDA, respiration, skin temp, EMG (1000 Hz) | continuous joystick valence/arousal | Video watching. [paper](https://www.nature.com/articles/s41597-019-0209-0) |
| **WESAD** | 15 | E4 + chest sensor | stress / amusement / baseline conditions | Condition labels rather than ratings |
| **EEVR** | 37 | EDA, PPG | valence/arousal + text descriptions | VR videos. [site](https://melangelabiiitd.github.io/EEVR/) |
| DEAP, AMIGOS, DREAMER, MAHNOB-HCI | 23–40 | EEG-centred + peripheral | valence/arousal per clip | Classic video/music-stimulus datasets; EULA required |

## Pretrained models

| Model | Input | Output | Reported on new people | Access |
|---|---|---|---|---|
| EEVR / FEEL "CLSP" | EDA + PPG | low/high valence and arousal, 4 quadrants | 0.72–0.81 F1 across datasets (0.73 when tested on E4 data) | [HF: Pragya/EEVR](https://huggingface.co/Pragya/EEVR) (gated, MIT), [FEEL](https://alchemy18.github.io/FEEL_Benchmark/) |
| UME (EDA foundation model) | 60 s E4 EDA | representation | 0.61–0.63 balanced acc. (WESAD, low/high) | weights pending |
| audEERING wav2vec2-msp-dim | speech audio | continuous arousal / valence / dominance | built for speech valence | [HF](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) (CC BY-NC-SA) |
| Pulse-PPG, NormWear | PPG / multi-sensor | foundation representations | task-dependent | [Pulse-PPG](https://arxiv.org/pdf/2502.01108), [NormWear](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear) |

The FEEL benchmark (19 datasets) concludes that **hand-crafted features with simple models (Random Forest, LDA) are competitive** with deep models for EDA/PPG emotion recognition.

## Papers reporting very high K-EmoCon accuracy, and why

| Paper | Headline | What's behind it |
|---|---|---|
| Zitouni et al., IEEE JBHI 2023, *"LSTM-Modeling of Emotion Recognition Using Peripheral Physiological Signals in Naturalistic Conversations"* ([PDF](https://ic.kaist.ac.kr/publications/papers/zitouni2023lstmmodeling.pdf)) | ~95–97% valence and arousal | **Binary** labels with rating 3 counted as *high* (≈67% of arousal and ≈83% of valence windows are "high"). **Same people in train and test** (4-fold CV over pooled, 20 s-overlapping windows). Temporal smoothing applied to predictions **and to ground truth**. Best numbers use partner ratings. **Their own leave-one-subject-out results: arousal 67.9%, valence 76.0%**, no better than always predicting "high" (67% of arousal and 83% of valence windows in our 26-person K-EmoCon data) |
| HGR-RFAF, Engineering Applications of AI 2025 ([abstract](https://www.sciencedirect.com/science/article/abs/pii/S0952197625033779)) | 91.15% arousal / 93.03% valence | Full text paywalled; the abstract does not claim subject-independent testing. No code or weights found |

**Takeaway:** when comparing to published numbers, first check (1) the number of classes and where "neutral" goes, (2) whether the test people were in training, and (3) whether labels were smoothed or altered.
