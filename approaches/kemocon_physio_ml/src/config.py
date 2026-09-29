"""Paths and constants shared across the project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# K-EmoCon cannot be redistributed, so data lives outside git: <approach>/data by default,
# or wherever the KEMOCON_DATA_DIR environment variable points.
DATA = Path(os.environ.get("KEMOCON_DATA_DIR", ROOT / "data"))
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

WINDOW_S = 5            # K-EmoCon self-annotations cover 5-second intervals
CONTEXT_S = 30          # trailing context used by the *_30s features
BASELINE_S = 180        # last 3 min before the debate = per-person reference period
MIN_EDA_COVERAGE = 0.5  # a window needs >= 50% of its EDA samples to be kept

RANDOM_SEED = 42
N_TEST_PAIRS = 3        # hold out 3 debate pairs (= 6 people) as the test set

# Feature groups (letters match the discussion / docs).
FEATURE_GROUPS = {
    "A_eda_stats":  ["eda_mean", "eda_std", "eda_min", "eda_max", "eda_slope"],
    "B_eda_scr":    ["eda_scr_count", "eda_scr_amp"],
    "C_bvp":        ["bvp_p2p"],
    "D_e4_hr":      ["e4_hr_mean", "e4_hr_std", "e4_hr_slope"],
    "E_ibi_hrv":    ["ibi_mean", "ibi_sdnn", "ibi_rmssd", "ibi_count"],
    "F_temp":       ["temp_mean", "temp_slope"],
    "G_acc":        ["acc_mag_mean", "acc_mag_std"],
    "H_polar_hr":   ["polar_hr_mean", "polar_hr_std", "polar_hr_slope"],
    "I_eeg_bands":  ["eeg_delta", "eeg_theta", "eeg_lowAlpha", "eeg_highAlpha",
                     "eeg_lowBeta", "eeg_highBeta", "eeg_lowGamma", "eeg_midGamma"],
    "J_eeg_ratios": ["eeg_alpha_beta_ratio", "eeg_theta_beta_ratio"],
    "K_esense":     ["attention_mean", "meditation_mean"],
    "L_context30s": ["eda_mean_30s", "eda_slope_30s", "eda_scr_count_30s",
                     "e4_hr_mean_30s", "ibi_sdnn_30s", "ibi_rmssd_30s"],
}
ALL_FEATURES = [c for cols in FEATURE_GROUPS.values() for c in cols]

_E4 = ["A_eda_stats", "B_eda_scr", "C_bvp", "D_e4_hr", "E_ibi_hrv", "F_temp", "G_acc", "L_context30s"]
FEATURE_SETS = {
    "E4":     _E4,
    "E4+EEG": _E4 + ["I_eeg_bands", "J_eeg_ratios"],
    "ALL":    _E4 + ["I_eeg_bands", "J_eeg_ratios", "H_polar_hr", "K_esense"],
}


def features_of(feature_set: str) -> list[str]:
    return [c for g in FEATURE_SETS[feature_set] for c in FEATURE_GROUPS[g]]


TARGETS = ["valence", "arousal"]
CLASS_NAMES = ["low", "neutral", "high"]   # 1-2 -> low, 3 -> neutral, 4-5 -> high


def to_class(score: int) -> int:
    return 0 if score <= 2 else (1 if score == 3 else 2)
