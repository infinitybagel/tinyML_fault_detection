"""Make the logistic-regression result figures for the slides / paper.

Run from anywhere (paths are relative to the experiment folder):
    python scripts/make_figures.py            slide versions  -> figures/slides/
    PAPER=1 python scripts/make_figures.py    IEEE versions   -> figures/paper/

Writes PNGs to figures/slides/ (or figures/paper/):
  1_spectra.png             FFT spectrum of each condition (new +/-8 g dataset) vs normal
  2_confusion_matrices.png  binary + multiclass confusion matrices (shared split), held-out result
  3_feature_weights.png     logistic regression weights on the 9 shared features
  4_esp32_footprint.png     what the model costs on the ESP32, and ESP32-vs-Python agreement
  5_live_tests.png          % of windows correct in each live test on the ESP32
  6_live_timelines.png      fault probability over time when a fault was added mid-run
"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the experiment folder
DATA = os.path.join(HERE, "data")
RUNS = os.path.join(HERE, "results", "live_runs")
PAPER = os.environ.get("PAPER") == "1"          # PAPER=1: IEEE-sized versions in figures/paper/
OUT = os.path.join(HERE, "figures", "paper") if PAPER else os.path.join(HERE, "figures", "slides")
os.makedirs(OUT, exist_ok=True)

FS, WIN = 1000.0, 1000
SCALE_MS2 = 9.80665 / 4096.0                      # +/-8 g
FREQ_BANDS = [(5, 25), (25, 75), (75, 200), (200, 500)]
FEATURES = ["rms", "peak", "std", "kurtosis", "crest_factor",
            "band_5_25hz", "band_25_75hz", "band_75_200hz", "band_200_500hz"]
TRAIN = ["normal", "imbalance", "loose_mount", "blocked_airflow"]
NICE = {"normal": "Normal", "imbalance": "Imbalance", "loose_mount": "Loose mount",
        "blocked_airflow": "Blocked airflow", "heldout": "Held-out: blade rub"}

# Colors: one series color, a polarity pair, neutral greys, and a reserved alert color
BLUE, ORANGE, GREY, LIGHT, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#c3c2b7", "#0b0b0b", "#52514e"
ALERT = "#d03b3b"
plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "font.size": 11, "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GREY,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
if PAPER:
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "savefig.dpi": 300, "lines.markersize": 3,
                         "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})


def fs(n):
    """Explicit font sizes, scaled down for the paper versions."""
    return round(n * 0.7, 1) if PAPER else n


def size(slide, paper):
    return paper if PAPER else slide


def suptitle(fig, text, **kw):
    if not PAPER:                                  # in the paper the caption replaces the figure title
        fig.suptitle(text, **kw)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def run(pattern):
    """Newest saved live run matching the pattern (its per-window CSV)."""
    f = sorted(p for p in glob.glob(os.path.join(RUNS, pattern)) if "_vs_recorded_normal" not in p)[-1]
    return pd.read_csv(f)


# ---- data ------------------------------------------------------------------------------------
def magnitude(csv_path):
    c = pd.read_csv(csv_path).dropna()[["ax", "ay", "az"]].to_numpy(dtype=float)
    return np.sqrt(((c * SCALE_MS2) ** 2).sum(axis=1))


def extract(w):                                     # same math as features.py
    rms = np.sqrt(np.mean(w ** 2)); peak = np.max(np.abs(w))
    ac = w - w.mean(); mag = np.abs(np.fft.rfft(ac)) * 2.0 / len(ac)
    return [rms, peak, np.std(w), kurtosis(w), peak / rms] + [np.mean(mag[lo:hi] ** 2) for lo, hi in FREQ_BANDS]


sessions = []
for label in TRAIN + ["heldout"]:
    for i, f in enumerate(sorted(glob.glob(os.path.join(DATA, f"{label}_*.csv")))):
        meta = json.load(open(f[:-4] + ".json"))
        sessions.append({"label": label, "idx": i, "file": f, "rpm": meta["mean_fan_rpm"], "mag": magnitude(f)})

rows = []
for s in sessions:
    m = s["mag"]
    for w in range(len(m) // WIN):
        rows.append(dict(zip(FEATURES, extract(m[w * WIN:(w + 1) * WIN])), label=s["label"], idx=s["idx"],
                         session=os.path.basename(s["file"])))
feat = pd.DataFrame(rows)


# ---- 1. spectra ------------------------------------------------------------------------------
def mean_spectrum(mag):
    n = len(mag) // WIN
    w = mag[:n * WIN].reshape(n, WIN)
    w = w - w.mean(axis=1, keepdims=True)
    return (np.abs(np.fft.rfft(w, axis=1)) * 2.0 / WIN).mean(axis=0)


freqs = np.fft.rfftfreq(WIN, 1 / FS)
pick = {s["label"]: s for s in sessions if s["idx"] == 1}   # session 2 of each (normal 1 has a bump)
normal_spec = mean_spectrum(pick["normal"]["mag"])
spin = np.mean([s["rpm"] for s in sessions]) / 60.0

fig, axes = plt.subplots(5, 1, figsize=size((11, 12), (7.16, 5.6)), sharex=True, sharey=True)
for ax, label in zip(axes, TRAIN + ["heldout"]):
    spec = mean_spectrum(pick[label]["mag"])
    if label != "normal":
        ax.semilogy(freqs[1:], normal_spec[1:] * 1000, color=LIGHT, lw=1.2, label="Normal (reference)")
    ax.semilogy(freqs[1:], spec[1:] * 1000, color=BLUE, lw=1.1, label=NICE[label])
    ax.axvline(spin, color=GREY, ls="--", lw=0.9)
    ax.axvline(7 * spin, color=GREY, ls=":", lw=0.9)
    ax.set_title(NICE[label], loc="left", fontweight="bold")
    ax.set_ylabel("mm/s²")
    ax.grid(axis="x", visible=False)
    if label != "normal":
        ax.legend(loc="upper right", frameon=False, fontsize=fs(9))
axes[0].text(spin + 3, 0.97, f"1× spin ≈ {spin:.0f} Hz", transform=axes[0].get_xaxis_transform(),
             va="top", color=MUTED, fontsize=fs(9))
axes[0].text(7 * spin + 3, 0.97, f"blade pass (7×) ≈ {7 * spin:.0f} Hz", transform=axes[0].get_xaxis_transform(),
             va="top", color=MUTED, fontsize=fs(9))
axes[-1].set_xlim(0, 500)
axes[-1].set_xlabel("Frequency (Hz)")
suptitle(fig, f"Vibration spectrum by condition (±8 g, fan at ~{spin * 60:.0f} RPM, mean of 1-s windows)",
             x=0.01, ha="left", fontsize=fs(13))
fig.tight_layout()
save(fig, "1_spectra.png")


# ---- model (same as the notebook: shared split = last session of each condition is test) ----
tr_df = feat[feat.label.isin(TRAIN)]
X = tr_df[FEATURES].to_numpy(); groups = tr_df.session.to_numpy()
y_bin = (tr_df.label != "normal").astype(int).to_numpy(); y_multi = tr_df.label.to_numpy()
test = (tr_df.idx == 4).to_numpy()
X_held = feat[feat.label == "heldout"][FEATURES].to_numpy()


def tuned(y, scoring):
    g = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=10000)),
                     {"logisticregression__C": np.logspace(-3, 3, 7)}, cv=GroupKFold(4), scoring=scoring, n_jobs=-1)
    g.fit(X[~test], y[~test], groups=groups[~test])
    return g.best_estimator_


bin_model, multi_model = tuned(y_bin, "f1"), tuned(y_multi, "f1_macro")


# ---- 2. confusion matrices -------------------------------------------------------------------
def plot_cm(ax, cm, names, title):
    ax.imshow(cm, cmap="Blues", vmin=0)
    ax.grid(False)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, loc="left", fontweight="bold")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else INK)


pred_bin = bin_model.predict(X[test])
pred_multi = multi_model.predict(X[test])
held_bin = bin_model.predict(X_held)
held_multi = pd.Series(multi_model.predict(X_held)).value_counts(normalize=True).reindex(TRAIN, fill_value=0) * 100

fig, axes = plt.subplots(1, 3, figsize=size((17, 5.2), (7.16, 2.55)), gridspec_kw={"width_ratios": [0.8, 1.2, 1.1]})
plot_cm(axes[0], confusion_matrix(y_bin[test], pred_bin, labels=[0, 1]), ["Normal", "Fault"],
        f"Normal vs. fault  (F1 = {f1_score(y_bin[test], pred_bin):.3f})")
plot_cm(axes[1], confusion_matrix(y_multi[test], pred_multi, labels=TRAIN), [NICE[l] for l in TRAIN],
        f"Which fault  (macro F1 = {f1_score(y_multi[test], pred_multi, average='macro'):.3f})")
ax = axes[2]
names = [NICE[l] for l in TRAIN][::-1]
vals = held_multi[TRAIN].to_numpy()[::-1]
ax.barh(names, vals, color=BLUE, height=0.55)
for i, v in enumerate(vals):
    ax.text(v + 1.5, i, f"{v:.0f}%", va="center", color=MUTED)
ax.set_xlim(0, 105); ax.set_xlabel("% of held-out windows")
ax.grid(axis="y", visible=False)
ax.set_title(f"Held-out rub ({held_bin.mean() * 100:.0f}% flagged)\nclassified as" if PAPER else
             f"Held-out blade rub: flagged as fault {held_bin.mean() * 100:.0f}%,\nclassified as …",
             loc="left", fontweight="bold")
suptitle(fig, "Logistic regression, offline results (test = last session of each condition; held-out never trained on)",
             x=0.01, ha="left", fontsize=fs(13))
fig.tight_layout()
save(fig, "2_confusion_matrices.png")


# ---- 3. feature weights ----------------------------------------------------------------------
def read_header(path=os.path.join(HERE, "firmware", "fault_detector", "logreg_model.h")):
    t = open(path).read()
    arr = lambda n: np.array([float(v.strip().rstrip("f")) for v in
                              re.search(n + r"\[N_FEATURES\] = \{(.*?)\}", t, re.S).group(1).split(",")])
    return arr("LR_MEAN"), arr("LR_SCALE"), arr("LR_COEF"), float(re.search(r"LR_INTERCEPT = ([^f;]+)f", t).group(1))


_, _, coef, _ = read_header()
w = pd.Series(coef, index=FEATURES).sort_values(key=np.abs)
fig, ax = plt.subplots(figsize=size((9, 5), (3.5, 2.5)))
ax.barh(w.index, w.values, color=[BLUE if v > 0 else ORANGE for v in w.values], height=0.6)
ax.axvline(0, color=GREY, lw=1)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Weight on standardized feature")
ax.set_title("Deployed model weights   (blue → fault, orange → normal)", loc="left", fontweight="bold")
ax.annotate("imbalance's own band\n(25–75 Hz) gets ~0 weight", xy=(w["band_25_75hz"], list(w.index).index("band_25_75hz")),
            xytext=(4, 1.2), color=MUTED, fontsize=fs(9), arrowprops=dict(arrowstyle="-", color=GREY))
fig.tight_layout()
save(fig, "3_feature_weights.png")


# ---- 4. ESP32 footprint ----------------------------------------------------------------------
ver = pd.read_csv(os.path.join(HERE, "results", "esp32_verify_results.csv"))
agree = (ver.esp_pred == ver.py_pred).mean() * 100
live_us = pd.concat([run(p) for p in ["live_normal_*.csv", "live_imbalance_*.csv", "live_loose_mount_*.csv",
                                      "live_blocked_airflow_midrun_*.csv"]]).inference_us.mean()
tiles = [
    ("112 B", "model parameters"),
    (f"{ver.infer_us.mean():.0f}–{live_us:.0f} µs", "inference per window\n(verify / live mode)"),
    (f"{ver.feat_us.mean() / 1000:.1f} ms", "feature extraction per 1-s window"),
    ("+8.7 KB", "flash vs. logger sketch"),
    ("+22 KB", "RAM vs. logger sketch\n(16 KB feature buffers)"),
    (f"{agree:.0f}%", f"ESP32 = Python predictions\n({len(ver)} recorded windows)"),
]
fig, axes = plt.subplots(2, 3, figsize=(12, 5))
for ax, (big, small) in zip(axes.ravel(), tiles):
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.02, 0.05), 0.96, 0.9, transform=ax.transAxes, fill=False, ec=LIGHT, lw=1.2))
    ax.text(0.5, 0.6, big, ha="center", va="center", fontsize=fs(30), color=INK, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.25, small, ha="center", va="center", fontsize=fs(11), color=MUTED, transform=ax.transAxes)
suptitle(fig, "Logistic regression on the ESP32 (ESP-WROOM-32, 240 MHz)", x=0.01, ha="left", fontsize=fs(14), fontweight="bold")
fig.tight_layout()
if PAPER:
    plt.close(fig)                                 # the paper uses a table for these numbers
else:
    save(fig, "4_esp32_footprint.png")


# ---- 5. live tests ---------------------------------------------------------------------------
static = [  # (label, file pattern, expected fault?)
    ("Normal", "live_normal_*.csv", 0),
    ("Imbalance", "live_imbalance_20261119_184150.csv", 1),
    ("Imbalance, weaker tie", "live_imbalance_20261119_183027.csv", 1),
    ("Loose mount, rattling", "live_loose_mount_20261119_185501.csv", 1),
    ("Loose mount, not rattling*", "live_loose_mount_20261119_185031.csv", 1),
]
res = []
for name, pat, want in static:
    d = run(pat)
    res.append((name, (d.pred == want).mean() * 100, len(d)))
fig, ax = plt.subplots(figsize=size((10, 4.3), (3.5, 2.3)))
names = [r[0] for r in res][::-1]; vals = [r[1] for r in res][::-1]; ns = [r[2] for r in res][::-1]
ax.barh(names, vals, color=BLUE, height=0.55)
for i, (v, n) in enumerate(zip(vals, ns)):
    ax.text(max(v, 0) + 1.5, i, f"{v:.0f}%  ({round(v * n / 100)}/{n}" + ("" if PAPER else " windows") + ")", va="center", color=MUTED)
ax.set_xlim(0, 125); ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("% of 1-s windows classified correctly on the ESP32")
ax.grid(axis="y", visible=False)
ax.set_title("Live tests on the ESP32 (30 s each)", loc="left", fontweight="bold")
None if PAPER else fig.text(0.01, -0.02, "* corner untaped but sitting still: vibration identical to normal, so there was nothing to detect",
         color=MUTED, fontsize=fs(9))
fig.tight_layout()
save(fig, "5_live_tests.png")


# ---- 6. live timelines (fault added mid-run) -------------------------------------------------
WINDOW_S = 1.0 + 0.0428       # 1 s of sampling + 42.8 ms of feature extraction
# Event times are window boundaries (window number x WINDOW_S), read from each run's scores.
timelines = [
    ("Blocked airflow: card placed at ~11.5 s*", "live_blocked_airflow_midrun_*.csv", [(11, "card placed")]),
    ("Loose mount: tape removed at ~10.5 s, rattle from ~18 s", "live_loose_mount_midrun_*.csv",
     [(10, "tape removed"), (17, "rattle starts")]),
    ("Blade rub: card clips blades at ~22 s", "live_fan_obstruction_midway_*.csv", [(21, "card touches blades")]),
]
fig, axes = plt.subplots(len(timelines), 1, figsize=size((11, 9), (7.16, 5.0)), sharex=True)
for ax, (title, pat, marks) in zip(axes, timelines):
    d = run(pat)
    t = (d.window + 0.5) * WINDOW_S             # middle of each 1-s window
    ax.plot(t, d.p_fault, color=BLUE, lw=size(2, 1.2), marker="o", ms=size(4, 2.5))
    ax.axhline(0.5, color=GREY, lw=0.9, ls="--")
    for w, txt in marks:
        x = w * WINDOW_S                        # start of the window where the change happened
        ax.axvline(x, color=GREY, lw=1, ls=":")
        ax.text(x - 0.3, 0.72, txt, color=MUTED, fontsize=fs(9), ha="right",
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
    for _, r in d[d.event == "ALERT"].iterrows():
        xa = (r.window + 1) * WINDOW_S          # the alert is sent at the end of its window
        ax.axvline(xa, color=ALERT, lw=1.5)
        ax.text(xa + 0.3, 0.3, ("" if PAPER else "⚠ ") + f"ALERT ({xa - marks[-1][0] * WINDOW_S:.1f} s after)", color=ALERT,
                fontsize=fs(10), fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_ylim(-0.05, 1.05); ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("P(fault)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", visible=False)
axes[-1].set_xlabel("Time since start (s)")
axes[-1].set_xlim(0, 42)
suptitle(fig, "Faults added while the ESP32 was running  (alert = 3 fault windows in a row)",
             x=0.01, ha="left", fontsize=fs(13))
None if PAPER else fig.text(0.01, -0.01, "* the card fluttered or lightly touched the blades: vibration was stronger than the recorded "
         "blocked-airflow condition", color=MUTED, fontsize=fs(9))
fig.tight_layout()
save(fig, "6_live_timelines.png")
