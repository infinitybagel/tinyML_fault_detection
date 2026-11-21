"""The shared evaluation. Every model is scored here, the same way.

    from benchmark import evaluate
    evaluate("linear_svm", make_model, kind="supervised", make_multiclass_model=..., device={...})

Split (fixed, never change it):
  * shared split:   train = sessions 1-4 of each training condition, test = session 5 (the last one)
  * 5 rounds:       round k tests on session k of every condition, trains on the other 4
  * held-out fault: all 5 'heldout' sessions, never used for training, only for testing

kind="supervised":   trained on windows of all 4 training conditions, label 1 = any fault.
                     Fault score = predict_proba(X)[:, 1] if available, else decision_function(X).
kind="unsupervised": trained ONLY on normal windows of the training sessions (scikit-learn outlier
                     convention: predict() returns -1 for an anomaly, decision_function() is higher
                     for more normal). Fault score = -decision_function(X).

Hyperparameter tuning must only use training sessions: wrap your estimator with tuned(...)
(GridSearchCV with GroupKFold over sessions); evaluate() passes the session groups automatically.

Writes to results/<name>/:
  <name>.json               all metrics (+ the on-device numbers you pass in `device`)
  <name>_predictions.csv    per-window predictions on the test + held-out windows (for the ESP32 check)
  <name>_confusion.png      binary (and multiclass) confusion matrices
"""

import inspect
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.model_selection._search import BaseSearchCV

from .features import FEATURES, HELDOUT, ROOT, TRAIN_LABELS, load_features

RESULTS = os.path.join(ROOT, "results")
DEVICE_FIELDS = ["param_bytes", "inference_us", "feature_ms", "flash_kb", "ram_kb", "esp32_agreement"]


def tuned(estimator, param_grid, scoring="f1", n_splits=4):
    """GridSearchCV that only ever splits by session (pass groups at fit time; evaluate() does)."""
    return GridSearchCV(estimator, param_grid, cv=GroupKFold(n_splits=n_splits), scoring=scoring, n_jobs=-1)


# ---- helpers ---------------------------------------------------------------------------------
def _fit(model, X, y, groups):
    """Fit, passing the session groups to anything that splits data internally (GridSearchCV etc.)."""
    params = inspect.signature(model.fit).parameters
    if isinstance(model, BaseSearchCV) or "groups" in params:
        model.fit(X, y, groups=groups)
    else:
        model.fit(X, y)
    return model


def _score_and_pred(model, X, kind):
    """(fault score, fault prediction 0/1) for any model."""
    if kind == "unsupervised":
        score = -model.decision_function(X)
        pred = (model.predict(X) == -1).astype(int)
    else:
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X)[:, 1]
        else:
            score = model.decision_function(X)
        pred = (np.asarray(model.predict(X)) == 1).astype(int)
    return np.asarray(score, dtype=float), pred


def _binary_metrics(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "false_alarm_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "roc_auc": roc_auc_score(y, score) if len(set(y)) > 1 else None,
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _round_split(df, k):
    """Train/test masks for the training conditions when session k is the test session."""
    train_conditions = df.label.isin(TRAIN_LABELS)
    return train_conditions & (df.session_idx != k), train_conditions & (df.session_idx == k)


def _fit_round(make_model, df, train, kind):
    sub = df[train]
    if kind == "unsupervised":
        sub = sub[sub.label == "normal"]                      # normal data only
    y = (sub.label != "normal").astype(int).to_numpy()
    return _fit(make_model(), sub[FEATURES].to_numpy(), y, sub.session.to_numpy())


# ---- main entry ------------------------------------------------------------------------------
def evaluate(name, make_model, kind="supervised", make_multiclass_model=None, device=None, owner="",
             notes="", rounds=True):
    """Train and score one model on the shared split (+5 rounds, held-out fault, optional multiclass).

    make_model:            function returning a NEW unfitted estimator (called once per split)
    kind:                  "supervised" or "unsupervised" (see module docstring)
    make_multiclass_model: optional, supervised only: estimator predicting the 4 training conditions
    device:                optional dict of ESP32 numbers, keys in DEVICE_FIELDS
    """
    assert kind in ("supervised", "unsupervised")
    out_dir = os.path.join(RESULTS, name)              # one folder per model
    os.makedirs(out_dir, exist_ok=True)
    df = load_features()
    held = df[df.label == HELDOUT]
    out = {"model": name, "owner": owner, "kind": kind, "notes": notes,
           "evaluated": datetime.now().isoformat(timespec="seconds"), "features": FEATURES}

    # 1) shared split: test = last session (index 4)
    train, test = _round_split(df, 4)
    model = _fit_round(make_model, df, train, kind)
    te = df[test]
    y_te = (te.label != "normal").astype(int).to_numpy()
    score, pred = _score_and_pred(model, te[FEATURES].to_numpy(), kind)
    h_score, h_pred = _score_and_pred(model, held[FEATURES].to_numpy(), kind)
    out["shared_split"] = _binary_metrics(y_te, pred, score)
    out["shared_split"]["heldout_recall"] = float(h_pred.mean())
    out["shared_split"]["n_test_windows"], out["shared_split"]["n_heldout_windows"] = int(len(te)), int(len(held))
    if hasattr(model, "best_params_"):
        out["chosen_params"] = {k: (v.item() if isinstance(v, np.generic) else
                                    v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                for k, v in model.best_params_.items()}

    preds = pd.concat([
        te[["session", "window", "label"]].assign(fault_score=score, pred=pred, truth=y_te),
        held[["session", "window", "label"]].assign(fault_score=h_score, pred=h_pred, truth=1),
    ])
    preds.to_csv(os.path.join(out_dir, f"{name}_predictions.csv"), index=False)

    # 2) five session-level rounds
    if rounds:
        per_round = []
        for k in range(5):
            tr_k, te_k = _round_split(df, k)
            m = _fit_round(make_model, df, tr_k, kind)
            t = df[te_k]
            s, p = _score_and_pred(m, t[FEATURES].to_numpy(), kind)
            _, hp = _score_and_pred(m, held[FEATURES].to_numpy(), kind)
            r = _binary_metrics((t.label != "normal").astype(int).to_numpy(), p, s)
            r["heldout_recall"] = float(hp.mean())
            r.pop("confusion")
            per_round.append(r)
        keys = ["accuracy", "precision", "recall", "f1", "false_alarm_rate", "roc_auc", "heldout_recall"]
        out["five_rounds"] = {k: {"mean": float(np.mean([r[k] for r in per_round if r[k] is not None])),
                                  "std": float(np.std([r[k] for r in per_round if r[k] is not None], ddof=1))}
                              for k in keys}

    # 3) which fault (supervised models, optional)
    mc_cm = None
    if make_multiclass_model is not None and kind == "supervised":
        tr = df[train]
        mm = _fit(make_multiclass_model(), tr[FEATURES].to_numpy(), tr.label.to_numpy(), tr.session.to_numpy())
        mp = mm.predict(te[FEATURES].to_numpy())
        mc_cm = confusion_matrix(te.label, mp, labels=TRAIN_LABELS)
        hp = pd.Series(mm.predict(held[FEATURES].to_numpy())).value_counts(normalize=True)
        out["multiclass"] = {
            "macro_f1": f1_score(te.label, mp, average="macro"),
            "accuracy": accuracy_score(te.label, mp),
            "labels": TRAIN_LABELS,
            "confusion": mc_cm.tolist(),
            "heldout_classified_as": {k: float(hp.get(k, 0.0)) for k in TRAIN_LABELS},
        }

    # 4) on-device numbers (fill in after testing on the ESP32)
    out["device"] = {k: (device or {}).get(k) for k in DEVICE_FIELDS}

    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    _plot_confusion(name, out["shared_split"]["confusion"], mc_cm)
    _print_summary(out)
    return out


def _plot_confusion(name, binary_cm, mc_cm):
    n = 2 if mc_cm is not None else 1
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5))
    axes = np.atleast_1d(axes)
    panels = [(np.array(binary_cm), ["normal", "fault"], "Normal vs. fault")]
    if mc_cm is not None:
        panels.append((mc_cm, TRAIN_LABELS, "Which fault"))
    for ax, (cm, names, title) in zip(axes, panels):
        ax.imshow(cm, cmap="Blues", vmin=0)
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{name}: {title}", loc="left")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, name, f"{name}_confusion.png"), dpi=150)
    plt.close(fig)


def _print_summary(out):
    s = out["shared_split"]
    print(f"\n=== {out['model']} ({out['kind']}) ===")
    print(f"Shared split:  precision {s['precision']:.3f}  recall {s['recall']:.3f}  F1 {s['f1']:.3f}  "
          f"false alarms {s['false_alarm_rate']:.3f}  ROC-AUC {s['roc_auc']:.3f}  held-out recall {s['heldout_recall']:.3f}")
    if "five_rounds" in out:
        r = out["five_rounds"]
        print(f"5 rounds:      F1 {r['f1']['mean']:.3f} ± {r['f1']['std']:.3f}  "
              f"false alarms {r['false_alarm_rate']['mean']:.3f}  held-out recall {r['heldout_recall']['mean']:.3f}")
    if "multiclass" in out:
        print(f"Which fault:   macro F1 {out['multiclass']['macro_f1']:.3f}")
    if "chosen_params" in out:
        print(f"Chosen params: {out['chosen_params']}")
    missing = [k for k, v in out["device"].items() if v is None]
    if missing:
        print(f"On-device numbers still missing: {', '.join(missing)}")
    print(f"Saved to results/{out['model']}/: .json, _predictions.csv, _confusion.png")
