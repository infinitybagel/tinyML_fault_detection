"""Compare every model that has been evaluated (results/*.json).

    python compare.py

Writes to results/comparison/:
  comparison_table.csv / .md   the paper's Table II (one row per model)
  f1_per_model.png             F1 on the shared split, with the 5-round mean ± std
  heldout_recall.png           recall on the held-out blade rub
  f1_vs_size.png               F1 vs. model size (needs device.param_bytes)
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "comparison")
os.makedirs(OUT, exist_ok=True)

ORDER = ["logistic_regression", "linear_svm", "isolation_forest", "isolation_forest_exported", "decision_tree"]
NICE = {"logistic_regression": "Logistic regression", "linear_svm": "Linear SVM",
        "isolation_forest": "Isolation forest", "isolation_forest_exported": "Isolation forest (exported)", "decision_tree": "Decision tree"}
BLUE, GREY, INK, MUTED = "#2a78d6", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e1", "axes.axisbelow": True, "font.size": 11,
                     "axes.edgecolor": GREY, "xtick.color": MUTED, "ytick.color": MUTED})

results = {}
for f in glob.glob(os.path.join(ROOT, "results", "*", "*.json")) + glob.glob(os.path.join(ROOT, "results", "*.json")):
    r = json.load(open(f))
    if "shared_split" in r:
        results[r["model"]] = r
names = [n for n in ORDER if n in results] + sorted(n for n in results if n not in ORDER)
if not names:
    raise SystemExit("No results yet. Run a model first, e.g.  python models/logistic_regression.py")

rows = []
for n in names:
    r, s, d = results[n], results[n]["shared_split"], results[n]["device"]
    fr = r.get("five_rounds", {})
    rows.append({
        "model": NICE.get(n, n), "owner": r.get("owner", ""), "kind": r["kind"],
        "precision": s["precision"], "recall": s["recall"], "f1": s["f1"],
        "false_alarm_rate": s["false_alarm_rate"], "roc_auc": s["roc_auc"], "heldout_recall": s["heldout_recall"],
        "f1_5round_mean": fr.get("f1", {}).get("mean"), "f1_5round_std": fr.get("f1", {}).get("std"),
        "multiclass_macro_f1": r.get("multiclass", {}).get("macro_f1"),
        **{k: d.get(k) for k in ["param_bytes", "inference_us", "flash_kb", "ram_kb", "esp32_agreement"]},
    })
table = pd.DataFrame(rows)
table.round(3).to_csv(os.path.join(OUT, "comparison_table.csv"), index=False)
md = table.round(3).astype(object).where(table.notna(), "—")
with open(os.path.join(OUT, "comparison_table.md"), "w", encoding="utf-8") as f:
    f.write("| " + " | ".join(md.columns) + " |\n")
    f.write("|" + "---|" * len(md.columns) + "\n")
    for _, row in md.iterrows():
        f.write("| " + " | ".join(str(v) for v in row) + " |\n")
print(table.round(3).fillna("—").to_string(index=False))

labels = [NICE.get(n, n) for n in names]


def bar(values, title, xlabel, fname, means=None, stds=None):
    """Horizontal bars with the value inside each bar; optional 5-round mean ± std as a grey marker."""
    fig, ax = plt.subplots(figsize=(8.5, 0.75 * len(labels) + 1.5))
    y = list(range(len(labels)))[::-1]
    ax.barh(y, values, color=BLUE, height=0.55)
    ax.set_yticks(y, labels)
    for yi, v in zip(y, values):
        ax.text(v - 0.015, yi, f"{v:.3f}", color="white", fontsize=10, va="center", ha="right", fontweight="bold")
    if means is not None:
        for yi, m, s in zip(y, means, stds):
            if m == m and m is not None:
                ax.errorbar(m, yi + 0.36, xerr=s if s == s else 0, fmt="o", color=INK, ms=4, capsize=3, lw=1)
        ax.plot([], [], "o", color=INK, ms=4, label="5-round mean ± std")
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9)
    ax.set_xlim(0, 1.05); ax.set_xlabel(xlabel); ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=200); plt.close(fig)


bar(table.f1.tolist(), "F1, normal vs. fault", "F1 on the shared test split (bars)", "f1_per_model.png",
    means=table.f1_5round_mean.tolist(), stds=table.f1_5round_std.tolist())
bar(table.heldout_recall.tolist(), "Recall on the held-out blade rub (never trained on)",
    "Fraction of held-out windows flagged as fault", "heldout_recall.png")

sized = table.dropna(subset=["param_bytes"])
if len(sized):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(sized.param_bytes, sized.f1, s=70, color=BLUE, zorder=3)
    for (size, f1), grp in sized.groupby(["param_bytes", "f1"]):          # models on the same point share a label
        ax.annotate(" / ".join(grp.model), (size, f1), xytext=(8, 6), textcoords="offset points", color=INK)
    ax.set_xlim(sized.param_bytes.min() / 2, sized.param_bytes.max() * 8)
    ax.set_xscale("log"); ax.set_xlabel("Model size on the ESP32 (bytes, log scale)"); ax.set_ylabel("F1")
    ax.set_ylim(min(0.9, sized.f1.min() - 0.02), 1.01)
    ax.set_title("F1 vs. model size", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "f1_vs_size.png"), dpi=200); plt.close(fig)
else:
    print("No model has device.param_bytes yet, so f1_vs_size.png was skipped.")
print(f"\nSaved to {OUT}")
