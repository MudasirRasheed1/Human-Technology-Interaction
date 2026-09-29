"""Train and evaluate valence/arousal classifiers with an abstain-when-unsure option.

Two deployment scenarios are compared, all decisions made by cross-validation on the
TRAIN people only (leave-one-debate-pair-out), then evaluated ONCE on the TEST people:

  GENERIC        new person, no information about them. Features may be normalised
                 against the person's pre-conversation baseline or their first 3 minutes.
  PERSONALISED   the first 3 minutes (36 windows) of the conversation are a calibration
                 phase in which the person also self-reports. Their rating tendencies
                 (class frequencies) are combined with the generic model's output:
                     p(class | sensors, person) ~ p_model(class | sensors) * prior_person(class)
                 Evaluated only on windows AFTER the calibration phase.
  PRIOR-ONLY     reference for PERSONALISED: ignore the sensors, always answer with the
                 person's calibration class frequencies. If the sensors add nothing,
                 PERSONALISED will not beat this.

Selective prediction: confidence = highest class probability. Thresholds are chosen on
cross-validated predictions to hit a target accuracy, then applied unchanged to TEST.

Usage:  python train.py            (reuses the cached CV grid if present)
        python train.py --regrid   (recompute the grid, ~20 min)
"""
import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (PROCESSED, MODELS, REPORTS, FIGURES, ALL_FEATURES, FEATURE_SETS, TARGETS,
                    CLASS_NAMES, RANDOM_SEED, features_of)
from features import stats_from_baseline, stats_from_calibration, normalise

sns.set_theme(style="whitegrid", context="notebook")
CACHE = REPORTS / "cache"
CALIB_WINDOWS = 36                    # first 3 minutes of the conversation
NORMS = ["none", "baseline", "calib"]
MODEL_NAMES = ["logreg", "random_forest", "lightgbm"]
TARGET_ACCS = [0.6, 0.7, 0.8]
DEFAULT_TARGET_ACC = 0.7
MIN_COVERAGE = 0.05
PRIOR_SMOOTHING = 1.0                 # Laplace smoothing of calibration class counts


def make_model(name):
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000))
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=300, min_samples_leaf=10, max_features="sqrt",
                                                    class_weight="balanced_subsample", n_jobs=-1,
                                                    random_state=RANDOM_SEED))
    if name == "lightgbm":
        return LGBMClassifier(n_estimators=250, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                              class_weight="balanced", random_state=RANDOM_SEED, verbose=-1)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
class Normaliser:
    """Per-person normalisation; the floor constant is learned from the fitting people only."""

    def __init__(self, norm, feats, ref_stats, fit_pids):
        self.norm, self.feats = norm, feats
        if norm == "none":
            return
        self.mu, self.sd = ref_stats[norm]
        fl = self.sd.loc[self.sd.index.isin(fit_pids), feats].median()
        self.floor = fl.where(np.isfinite(fl) & (fl > 0), 1e-6)

    def __call__(self, df):
        return df[self.feats] if self.norm == "none" else normalise(df, self.feats, self.mu, self.sd, self.floor)


def fit_predict(fit_df, apply_df, target, fs, norm, mn, ref_stats):
    feats = features_of(fs)
    nz = Normaliser(norm, feats, ref_stats, fit_df.pid.unique())
    model = make_model(mn).fit(nz(fit_df), fit_df[f"{target}_cls"].to_numpy())
    return model, nz, model.predict_proba(nz(apply_df))


def oof_generic(train, target, fs, norm, mn, ref_stats):
    proba = np.zeros((len(train), 3))
    for tr_idx, va_idx in LeaveOneGroupOut().split(train, groups=train.pair):
        _, _, proba[va_idx] = fit_predict(train.iloc[tr_idx], train.iloc[va_idx], target, fs, norm, mn, ref_stats)
    return proba


# ---------------------------------------------------------------------------
# Personalisation
# ---------------------------------------------------------------------------
def person_priors(df, target):
    """Class frequencies of each person's calibration windows (Laplace-smoothed), indexed by pid."""
    cal = df[df.calib]
    counts = pd.crosstab(cal.pid, cal[f"{target}_cls"]).reindex(columns=[0, 1, 2], fill_value=0)
    return (counts + PRIOR_SMOOTHING).div(counts.sum(1) + 3 * PRIOR_SMOOTHING, axis=0)


def personalise(proba, df, priors):
    p = proba * priors.reindex(df.pid).to_numpy()
    return p / p.sum(1, keepdims=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0))}


def per_group_f1(y, pred, groups):
    s = [f1_score(y[groups == g], pred[groups == g], average="macro", labels=[0, 1, 2], zero_division=0)
         for g in np.unique(groups)]
    return float(np.std(s))


def risk_coverage(y, proba):
    conf, pred = proba.max(1), proba.argmax(1)
    order = np.argsort(-conf, kind="stable")
    correct = (pred[order] == np.asarray(y)[order]).astype(float)
    n = np.arange(1, len(y) + 1)
    return n / len(y), np.cumsum(correct) / n, conf[order]


def acc_at_coverage(y, proba, c):
    cov, acc, _ = risk_coverage(y, proba)
    return float(acc[min(np.searchsorted(cov, c), len(acc) - 1)])


def pick_threshold(y, proba, target):
    cov, acc, thr = risk_coverage(y, proba)
    ok = np.where((acc >= target) & (cov >= MIN_COVERAGE))[0]
    return None if len(ok) == 0 else float(thr[ok.max()])


def selective(y, proba, thr):
    if thr is None:
        return None
    y = np.asarray(y)
    keep = proba.max(1) >= thr
    if keep.sum() == 0:
        return {"threshold": thr, "coverage": 0.0, "n_accepted": 0, "accuracy": None, "balanced_accuracy": None}
    pred = proba.argmax(1)
    return {"threshold": thr, "coverage": float(keep.mean()), "n_accepted": int(keep.sum()),
            "accuracy": float(accuracy_score(y[keep], pred[keep])),
            "balanced_accuracy": float(balanced_accuracy_score(y[keep], pred[keep])),
            "predicted_class_mix": np.bincount(pred[keep], minlength=3).tolist()}


def summarise(y, proba, groups):
    pred = proba.argmax(1)
    return {**metrics(y, pred), "macro_f1_std_across_pairs": per_group_f1(y, pred, groups),
            "acc_at_25pct": acc_at_coverage(y, proba, 0.25), "acc_at_50pct": acc_at_coverage(y, proba, 0.5)}


# ---------------------------------------------------------------------------
def load():
    train = pd.read_csv(PROCESSED / "train.csv")
    test = pd.read_csv(PROCESSED / "test.csv")
    for df in (train, test):
        df.sort_values(["pid", "window_end_s"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["calib"] = df.groupby("pid").cumcount() < CALIB_WINDOWS
    base = pd.read_csv(PROCESSED / "baseline_stats.csv").set_index("pid")
    ref_stats = {"baseline": stats_from_baseline(base, ALL_FEATURES),
                 "calib": stats_from_calibration(pd.concat([train, test]), ALL_FEATURES, CALIB_WINDOWS)}
    return train, test, ref_stats


def run_grid(train, ref_stats):
    rows, oof = [], {}
    t0 = time.time()
    for target in TARGETS:
        y = train[f"{target}_cls"].to_numpy()
        for fs in FEATURE_SETS:
            for norm in NORMS:
                for mn in MODEL_NAMES:
                    p = oof_generic(train, target, fs, norm, mn, ref_stats)
                    oof[(target, fs, norm, mn)] = p
                    rows.append({"target": target, "feature_set": fs, "norm": norm, "model": mn,
                                 **summarise(y, p, train.pair.to_numpy())})
                    r = rows[-1]
                    print(f"[{time.time() - t0:5.0f}s] {target:7s} {fs:6s} {norm:8s} {mn:13s} macroF1={r['macro_f1']:.3f}"
                          f" (+-{r['macro_f1_std_across_pairs']:.3f}) acc@25%={r['acc_at_25pct']:.3f}", flush=True)
    return pd.DataFrame(rows), oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regrid", action="store_true")
    args = ap.parse_args()
    for d in (MODELS, FIGURES, CACHE):
        d.mkdir(parents=True, exist_ok=True)

    train, test, ref_stats = load()
    cache = CACHE / "grid.joblib"
    if cache.exists() and not args.regrid:
        grid, oof = joblib.load(cache)
        print("loaded cached CV grid")
    else:
        grid, oof = run_grid(train, ref_stats)
        joblib.dump((grid, oof), cache)
    grid.to_csv(REPORTS / "cv_grid_generic.csv", index=False)

    ev_tr = ~train.calib.to_numpy()          # post-calibration windows
    ev_te = ~test.calib.to_numpy()
    results = {"n_train_rows": len(train), "n_test_rows": len(test),
               "n_test_rows_after_calibration": int(ev_te.sum()),
               "train_pids": sorted(train.pid.unique().tolist()), "test_pids": sorted(test.pid.unique().tolist()),
               "calibration_windows": CALIB_WINDOWS, "targets": {}}
    arrays = {}

    for target in TARGETS:
        yc = f"{target}_cls"
        y_tr, y_te = train[yc].to_numpy(), test[yc].to_numpy()
        g_tr, g_te = train.pair.to_numpy(), test.pair.to_numpy()
        pri_tr, pri_te = person_priors(train, target), person_priors(test, target)
        R = {}

        # ---- CV comparison of approaches on post-calibration windows ----------
        majority = np.bincount(y_tr).argmax()
        cv_rows = []
        for (t, fs, norm, mn), p in oof.items():
            if t != target:
                continue
            cv_rows.append({"approach": "generic", "feature_set": fs, "norm": norm, "model": mn,
                            **summarise(y_tr[ev_tr], p[ev_tr], g_tr[ev_tr])})
            pp = personalise(p, train, pri_tr)
            cv_rows.append({"approach": "personalised", "feature_set": fs, "norm": norm, "model": mn,
                            **summarise(y_tr[ev_tr], pp[ev_tr], g_tr[ev_tr])})
        prior_only_tr = pri_tr.reindex(train.pid).to_numpy()
        cv_rows.append({"approach": "prior_only", "feature_set": "-", "norm": "-", "model": "-",
                        **summarise(y_tr[ev_tr], prior_only_tr[ev_tr], g_tr[ev_tr])})
        maj_p = np.tile(np.eye(3)[majority], (len(y_tr), 1))
        cv_rows.append({"approach": "majority_class", "feature_set": "-", "norm": "-", "model": "-",
                        **summarise(y_tr[ev_tr], maj_p[ev_tr], g_tr[ev_tr])})
        cv = pd.DataFrame(cv_rows)
        cv.to_csv(REPORTS / f"cv_approaches_{target}.csv", index=False)

        best = {a: cv[cv.approach == a].sort_values("macro_f1", ascending=False).iloc[0]
                for a in ["generic", "personalised"]}
        R["cv_best"] = {a: best[a].to_dict() for a in best}
        R["cv_prior_only"] = cv[cv.approach == "prior_only"].iloc[0].to_dict()
        R["cv_majority"] = cv[cv.approach == "majority_class"].iloc[0].to_dict()
        cand = {"generic": best["generic"].macro_f1, "personalised": best["personalised"].macro_f1,
                "prior_only": R["cv_prior_only"]["macro_f1"]}
        recommended = max(cand, key=cand.get)
        R["recommended_by_cv"] = recommended

        # ---- Fit on all train people, evaluate on test -------------------------
        te = {}
        for a in ["generic", "personalised"]:
            b = best[a]
            model, nz, p_te = fit_predict(train, test, target, b.feature_set, b.norm, b.model, ref_stats)
            oof_p = oof[(target, b.feature_set, b.norm, b.model)]
            if a == "personalised":
                p_te, oof_p = personalise(p_te, test, pri_te), personalise(oof_p, train, pri_tr)
            te[a] = {"model": model, "nz": nz, "p_te": p_te, "oof": oof_p, "cfg": b}
        te["prior_only"] = {"p_te": pri_te.reindex(test.pid).to_numpy(), "oof": prior_only_tr, "cfg": None}

        R["test"] = {}
        for a, d in te.items():
            thr = {str(t): pick_threshold(y_tr[ev_tr], d["oof"][ev_tr], t) for t in TARGET_ACCS}
            d["thr"] = thr
            R["test"][a] = {
                "after_calibration": summarise(y_te[ev_te], d["p_te"][ev_te], g_te[ev_te]),
                "selective_after_calibration": {t: selective(y_te[ev_te], d["p_te"][ev_te], v) for t, v in thr.items()},
                "cv_selective_after_calibration": {t: selective(y_tr[ev_tr], d["oof"][ev_tr], v) for t, v in thr.items()},
                "thresholds": thr,
            }
            if a == "generic":
                R["test"][a]["all_windows"] = summarise(y_te, d["p_te"], g_te)
        R["test_majority_acc_after_calibration"] = float((y_te[ev_te] == majority).mean())

        # ---- Per-participant + confusion for the recommended approach ----------
        rec = te[recommended]
        thr = rec["thr"][str(DEFAULT_TARGET_ACC)]
        pred = rec["p_te"].argmax(1)
        conf = rec["p_te"].max(1)
        R["test_confusion_all"] = confusion_matrix(y_te[ev_te], pred[ev_te], labels=[0, 1, 2]).tolist()
        k = ev_te & (conf >= thr) if thr is not None else np.zeros_like(ev_te)
        if k.any():
            R["test_confusion_confident"] = confusion_matrix(y_te[k], pred[k], labels=[0, 1, 2]).tolist()
        per = []
        for pid in sorted(test.pid.unique()):
            m = ev_te & (test.pid == pid).to_numpy()
            r = {"pid": int(pid), "n": int(m.sum()), "acc_all": float((pred[m] == y_te[m]).mean()),
                 "acc_prior_only": float((te["prior_only"]["p_te"][m].argmax(1) == y_te[m]).mean())}
            if thr is not None:
                k = m & (conf >= thr)
                r["coverage"] = float(k.sum() / m.sum())
                r["acc_confident"] = float((pred[k] == y_te[k]).mean()) if k.any() else None
            per.append(r)
        R["test_per_participant"] = per

        # ---- Permutation importance of the sensor model behind the personalised approach
        b = best["personalised"]
        feats = features_of(b.feature_set)
        imps = []
        for tr_idx, va_idx in LeaveOneGroupOut().split(train, groups=train.pair):
            tr, va = train.iloc[tr_idx], train.iloc[va_idx]
            model, nz, _ = fit_predict(tr, va.iloc[:1], target, b.feature_set, b.norm, b.model, ref_stats)
            pi = permutation_importance(model, nz(va), va[yc].to_numpy(), scoring="f1_macro",
                                        n_repeats=3, random_state=RANDOM_SEED, n_jobs=1)
            imps.append(pi.importances_mean)
        R["permutation_importance"] = dict(zip(feats, np.round(np.mean(imps, axis=0), 5).tolist()))

        # ---- Save deployable artefacts ----------------------------------------
        for a in ["generic", "personalised"]:
            d = te[a]
            joblib.dump({"model": d["model"], "normaliser": d["nz"], "features": features_of(d["cfg"].feature_set),
                         "config": {k: d["cfg"][k] for k in ["feature_set", "norm", "model"]},
                         "approach": a, "target": target, "class_names": CLASS_NAMES,
                         "thresholds": d["thr"], "default_target_accuracy": DEFAULT_TARGET_ACC,
                         "calibration_windows": CALIB_WINDOWS, "prior_smoothing": PRIOR_SMOOTHING},
                        MODELS / f"{target}_{a}.joblib")

        results["targets"][target] = R
        arrays[target] = {"y_tr": y_tr, "y_te": y_te, "ev_tr": ev_tr, "ev_te": ev_te, "te": te,
                          "recommended": recommended, "cv": cv}
        print(f"\n=== {target}: recommended by CV = {recommended}")
        for a in te:
            s = R["test"][a]
            print(f"  {a:12s} test(after calib) {s['after_calibration']}")
            print(f"  {'':12s} selective@0.7 {s['selective_after_calibration'][str(DEFAULT_TARGET_ACC)]}")

    make_figures(grid, results, arrays, test)
    json.dump(results, open(REPORTS / "results.json", "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("done ->", REPORTS)


# ---------------------------------------------------------------------------
def make_figures(grid, results, arrays, test):
    C = {"generic": "#8d99ae", "personalised": "#2e86ab", "prior_only": "#e09f3e", "majority": "#333333"}

    # 1. Generic CV grid
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    for ax, target in zip(axes, TARGETS):
        g = grid[grid.target == target].copy()
        g["config"] = g.feature_set + " | " + g.norm
        order = g.config.unique()
        w = 0.27
        for i, mn in enumerate(MODEL_NAMES):
            s = g[g.model == mn].set_index("config").reindex(order)
            ax.bar(np.arange(len(order)) + (i - 1) * w, s.macro_f1, w, yerr=s.macro_f1_std_across_pairs,
                   capsize=2, label=mn, color=sns.color_palette("Set2")[i], error_kw={"lw": 0.7})
        ax.axhline(1 / 3, ls=":", color="grey", lw=1, label="chance (1/3)")
        ax.set_xticks(range(len(order)), order, rotation=60, fontsize=8, ha="right")
        ax.set_title(f"{target.capitalize()}: GENERIC model, leave-one-pair-out CV\n(error bar = spread across held-out pairs)")
        ax.set_ylabel("macro F1"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "model_cv_generic_grid.png", dpi=150); plt.close(fig)

    # 2. Approach comparison, CV vs TEST
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, target in zip(axes, TARGETS):
        R = results["targets"][target]
        names = ["majority", "generic", "prior_only", "personalised"]
        cvv = [R["cv_majority"]["macro_f1"], R["cv_best"]["generic"]["macro_f1"], R["cv_prior_only"]["macro_f1"],
               R["cv_best"]["personalised"]["macro_f1"]]
        maj_te = arrays[target]["y_te"][arrays[target]["ev_te"]]
        maj_cls = np.bincount(arrays[target]["y_tr"]).argmax()
        tev = [f1_score(maj_te, np.full_like(maj_te, maj_cls), average="macro", labels=[0, 1, 2], zero_division=0),
               R["test"]["generic"]["after_calibration"]["macro_f1"], R["test"]["prior_only"]["after_calibration"]["macro_f1"],
               R["test"]["personalised"]["after_calibration"]["macro_f1"]]
        x = np.arange(len(names))
        ax.bar(x - 0.2, cvv, 0.4, label="CV (train people)", color=[C[n] for n in names], alpha=0.55)
        ax.bar(x + 0.2, tev, 0.4, label="TEST (held-out people)", color=[C[n] for n in names])
        for xi, (a, b) in enumerate(zip(cvv, tev)):
            ax.text(xi - 0.2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
            ax.text(xi + 0.2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax.axhline(1 / 3, ls=":", color="grey", lw=1)
        ax.set_xticks(x, ["majority\nclass", "generic\n(sensors)", "prior only\n(calib labels)", "personalised\n(sensors+calib)"])
        ax.set_title(f"{target.capitalize()}: macro F1 after the 3-min calibration\n(light = CV, dark = TEST)")
        ax.set_ylabel("macro F1"); ax.set_ylim(0, 0.8)
    fig.tight_layout(); fig.savefig(FIGURES / "model_approach_comparison.png", dpi=150); plt.close(fig)

    # 3. Risk-coverage on TEST for all approaches
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, target in zip(axes, TARGETS):
        A = arrays[target]
        y, ev = A["y_te"], A["ev_te"]
        for a in ["generic", "prior_only", "personalised"]:
            cov, acc, _ = risk_coverage(y[ev], A["te"][a]["p_te"][ev])
            ax.plot(cov, acc, label=a.replace("_", " "), color=C[a], lw=2)
            s = results["targets"][target]["test"][a]["selective_after_calibration"][str(DEFAULT_TARGET_ACC)]
            if s and s["accuracy"] is not None and s["coverage"] > 0:
                ax.scatter([s["coverage"]], [s["accuracy"]], color=C[a], s=60, zorder=5, edgecolor="k")
        ax.axhline(results["targets"][target]["test_majority_acc_after_calibration"], ls="--", color="k", lw=1,
                   label="always majority class")
        ax.axhline(DEFAULT_TARGET_ACC, ls=":", color="#d1495b", lw=1, label=f"target {DEFAULT_TARGET_ACC:.0%}")
        ax.set_xlabel("coverage (fraction of windows answered)"); ax.set_ylabel("accuracy on answered windows")
        ax.set_title(f"{target.capitalize()} - TEST: accuracy vs coverage\n(dot = operating point of the {DEFAULT_TARGET_ACC:.0%} threshold)")
        ax.set_ylim(0.2, 1.02); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "model_risk_coverage_test.png", dpi=150); plt.close(fig)

    # 4. Confusion matrices for the recommended approach
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    for i, target in enumerate(TARGETS):
        R = results["targets"][target]
        for j, (title, cm) in enumerate([("all windows", R["test_confusion_all"]),
                                         (f"confident only ({DEFAULT_TARGET_ACC:.0%} threshold)", R.get("test_confusion_confident"))]):
            ax = axes[i, j]
            if cm is None:
                ax.axis("off"); ax.text(0.5, 0.5, "target accuracy not reachable in CV", ha="center"); continue
            cm = np.array(cm)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
            ax.set_title(f"{target.capitalize()} [{R['recommended_by_cv']}] - {title}\nTEST, n={cm.sum()}", fontsize=10)
            ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.tight_layout(); fig.savefig(FIGURES / "model_confusion_test.png", dpi=150); plt.close(fig)

    # 5. Reliability of the confidence score (recommended approach)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bins = np.linspace(1 / 3, 1, 8)
    for ax, target in zip(axes, TARGETS):
        A = arrays[target]; rec = A["te"][A["recommended"]]
        for name, y, p, c in [("CV", A["y_tr"][A["ev_tr"]], rec["oof"][A["ev_tr"]], "#8d99ae"),
                              ("TEST", A["y_te"][A["ev_te"]], rec["p_te"][A["ev_te"]], "#2e86ab")]:
            conf, corr = p.max(1), p.argmax(1) == y
            idx = np.digitize(conf, bins) - 1
            pts = [(conf[idx == b].mean(), corr[idx == b].mean()) for b in range(len(bins) - 1) if (idx == b).sum() >= 10]
            if pts:
                ax.plot(*zip(*pts), "o-", color=c, label=name)
        ax.plot([1 / 3, 1], [1 / 3, 1], "k--", lw=1, label="ideal")
        ax.set_xlabel("confidence (max probability)"); ax.set_ylabel("actual accuracy")
        ax.set_title(f"{target.capitalize()} [{A['recommended']}]: is confidence trustworthy?"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "model_reliability.png", dpi=150); plt.close(fig)

    # 6. Feature importance
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for ax, target in zip(axes, TARGETS):
        imp = pd.Series(results["targets"][target]["permutation_importance"]).sort_values().tail(20)
        ax.barh(imp.index, imp.values, color=["#2e86ab" if v > 0 else "#bdbdbd" for v in imp.values])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{target.capitalize()}: top 20 features\n(drop in CV macro-F1 when shuffled)")
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "model_feature_importance.png", dpi=150); plt.close(fig)

    # 7. Per participant (recommended approach)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, target in zip(axes, TARGETS):
        per = pd.DataFrame(results["targets"][target]["test_per_participant"])
        x = np.arange(len(per))
        ax.bar(x - 0.27, per.acc_prior_only, 0.27, label="prior only", color=C["prior_only"])
        ax.bar(x, per.acc_all, 0.27, label=f"{arrays[target]['recommended']} (all)", color="#bdbdbd")
        if "acc_confident" in per:
            ax.bar(x + 0.27, per.acc_confident.fillna(0), 0.27, label="confident only", color=C["personalised"])
            for xi, c in zip(x, per.coverage):
                ax.text(xi + 0.27, 0.02, f"{c:.0%}", ha="center", fontsize=7, color="white", rotation=90)
        ax.set_xticks(x, [f"P{p}" for p in per.pid]); ax.set_ylim(0, 1)
        ax.set_title(f"{target.capitalize()}: TEST accuracy per person (after calibration)\nwhite = share of windows answered")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURES / "model_test_per_participant.png", dpi=150); plt.close(fig)

    # 8. Timeline for each test participant (recommended approach)
    pids = sorted(test.pid.unique())
    fig, axes = plt.subplots(len(pids), 2, figsize=(16, 2.1 * len(pids)), sharex=True)
    for j, target in enumerate(TARGETS):
        A = arrays[target]; rec = A["te"][A["recommended"]]
        thr = rec["thr"][str(DEFAULT_TARGET_ACC)]
        for i, pid in enumerate(pids):
            ax = axes[i, j]
            m = (test.pid == pid).to_numpy()
            t = test.window_end_s.to_numpy()[m] / 60
            y, p = A["y_te"][m], rec["p_te"][m]
            cal = test.calib.to_numpy()[m]
            ax.axvspan(t[cal].min(), t[cal].max(), color="#fff3b0", lw=0)
            ax.step(t, y, where="post", color="k", lw=1.3)
            conf, pred = p.max(1), p.argmax(1)
            ok = (~cal) & (conf >= thr) if thr is not None else np.zeros_like(cal)
            ax.scatter(t[ok], pred[ok] + 0.12, color="#2e86ab", s=12)
            ax.scatter(t[(~cal) & ~ok], pred[(~cal) & ~ok] + 0.12, color="#cccccc", s=6)
            ax.set_yticks([0, 1, 2], CLASS_NAMES, fontsize=7); ax.set_ylim(-0.4, 2.5)
            ax.set_ylabel(f"P{pid}", fontsize=9)
            if i == 0:
                ax.set_title(f"{target.capitalize()} [{A['recommended']}] - black: self-report, blue: confident prediction, "
                             f"grey: abstained, yellow: calibration", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("minutes into conversation")
    fig.tight_layout(); fig.savefig(FIGURES / "model_test_timelines.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
