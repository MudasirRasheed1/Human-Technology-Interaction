"""Descriptive charts of the prepared dataset -> reports/figures/data_*.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import PROCESSED, FIGURES, ALL_FEATURES, FEATURE_GROUPS, CLASS_NAMES
from features import session_normalise

sns.set_theme(style="whitegrid", context="notebook")
CLS_COLORS = ["#d1495b", "#bdbdbd", "#2e86ab"]


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED / "dataset.csv")
    test_pids = set(pd.read_csv(PROCESSED / "test.csv").pid.unique())

    # 1. Label distributions (1-5 and 3-class)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, t in enumerate(["valence", "arousal"]):
        counts = df[t].value_counts().reindex(range(1, 6), fill_value=0)
        colors = [CLS_COLORS[0]] * 2 + [CLS_COLORS[1]] + [CLS_COLORS[2]] * 2
        ax[i].bar(counts.index, counts.values, color=colors)
        for x, y in zip(counts.index, counts.values):
            ax[i].text(x, y, f"{y}\n({y / len(df):.0%})", ha="center", va="bottom", fontsize=9)
        ax[i].set_title(f"{t.capitalize()} self-ratings (colour = 3-class bin)")
        ax[i].set_xlabel("rating (1-5)"); ax[i].set_ylabel("windows")
        ax[i].set_ylim(0, counts.max() * 1.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLS_COLORS]
    fig.legend(handles, ["low (1-2)", "neutral (3)", "high (4-5)"], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(); fig.savefig(FIGURES / "data_label_distribution.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # 2. Joint valence x arousal
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    joint = pd.crosstab(df.arousal, df.valence).reindex(index=range(5, 0, -1), columns=range(1, 6), fill_value=0)
    sns.heatmap(joint, annot=True, fmt="d", cmap="Blues", ax=ax, cbar_kws={"label": "windows"})
    ax.set_title("Valence x arousal (self-ratings)")
    fig.tight_layout(); fig.savefig(FIGURES / "data_valence_arousal_joint.png", dpi=150); plt.close(fig)

    # 3. Class mix per participant
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    pids = sorted(df.pid.unique())
    for i, t in enumerate(["valence_cls", "arousal_cls"]):
        frac = pd.crosstab(df.pid, df[t], normalize="index").reindex(index=pids, columns=[0, 1, 2], fill_value=0)
        bottom = np.zeros(len(pids))
        for k in range(3):
            ax[i].bar([str(p) for p in pids], frac[k], bottom=bottom, color=CLS_COLORS[k], label=CLASS_NAMES[k])
            bottom += frac[k].values
        ax[i].set_ylabel("fraction of windows"); ax[i].set_title(f"{t.split('_')[0].capitalize()} class mix per participant")
        for j, p in enumerate(pids):
            if p in test_pids:
                ax[i].text(j, 1.01, "test", ha="center", fontsize=8, color="black")
    ax[0].legend(ncol=3, loc="lower right"); ax[1].set_xlabel("participant id")
    n = df.pid.value_counts().reindex(pids)
    ax[1].set_xticks(range(len(pids)), [f"{p}\n(n={n[p]})" for p in pids], fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "data_labels_per_participant.png", dpi=150); plt.close(fig)

    # 4. Missing values per feature
    miss = df[ALL_FEATURES].isna().mean()
    group_of = {c: g for g, cols in FEATURE_GROUPS.items() for c in cols}
    fig, ax = plt.subplots(figsize=(12, 4.5))
    palette = dict(zip(FEATURE_GROUPS, sns.color_palette("tab20", len(FEATURE_GROUPS))))
    ax.bar(miss.index, miss.values, color=[palette[group_of[c]] for c in miss.index])
    ax.set_ylabel("fraction missing"); ax.set_title("Missing values per feature (colour = feature group)")
    ax.tick_params(axis="x", rotation=90, labelsize=8); ax.set_ylim(0, 1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[g]) for g in FEATURE_GROUPS]
    ax.legend(handles, list(FEATURE_GROUPS), fontsize=7, ncol=4, loc="upper left")
    fig.tight_layout(); fig.savefig(FIGURES / "data_missingness.png", dpi=150); plt.close(fig)

    # 5. Within-person feature/label correlation (features z-scored per person so that
    #    differences between people don't dominate).
    z = session_normalise(df, ALL_FEATURES)
    corr = pd.DataFrame({t: z[ALL_FEATURES].corrwith(df[t], method="spearman") for t in ["valence", "arousal"]})
    corr = corr.loc[corr.abs().max(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(5, 11))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-0.3, vmax=0.3, ax=ax, annot_kws={"size": 7})
    ax.set_title("Spearman correlation with rating\n(features z-scored within person)")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "data_feature_label_correlation.png", dpi=150); plt.close(fig)
    corr.round(4).to_csv(PROCESSED / "feature_label_correlation.csv")

    # 6. Example participant timeline
    pid = df.groupby("pid").arousal.std().idxmax()
    d = df[df.pid == pid].sort_values("window_end_s")
    t = d.window_end_s / 60
    panels = [("eda_mean", "EDA (uS)"), ("e4_hr_mean", "E4 heart rate (bpm)"), ("temp_mean", "skin temp (C)"),
              ("eeg_alpha_beta_ratio", "EEG log alpha/beta"), ("acc_mag_std", "movement (g std)")]
    fig, ax = plt.subplots(len(panels) + 1, 1, figsize=(12, 11), sharex=True)
    for a, (c, lab) in zip(ax, panels):
        a.plot(t, d[c], lw=1.2); a.set_ylabel(lab, fontsize=9)
    ax[-1].step(t, d.arousal, where="post", label="arousal", color="#e07a1f")
    ax[-1].step(t, d.valence, where="post", label="valence", color="#3a7d44")
    ax[-1].set_ylabel("self-rating"); ax[-1].set_yticks(range(1, 6)); ax[-1].legend(loc="upper right", ncol=2)
    ax[-1].set_xlabel("minutes into debate")
    ax[0].set_title(f"Participant {pid}: 5 s window features and self-ratings")
    fig.tight_layout(); fig.savefig(FIGURES / "data_example_timeline.png", dpi=150); plt.close(fig)
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
