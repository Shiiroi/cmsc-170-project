# https://matplotlib.org/stable/users/index
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from metrics import TIER_NAMES

TIERS = list(TIER_NAMES.values())

def plot_metric_bar(results, metric, out_path):
    methods = list(results.keys())
    x = np.arange(len(methods))
    width = 0.25

    figure, axis = plt.subplots(figsize=(8, 5))
    for i, tier in enumerate(TIERS):
        vals = [results[m][f"{metric}_by_class"][tier] for m in methods]
        axis.bar(x + i * width, vals, width, label=tier)

    axis.set_xticks(x + width)
    axis.set_xticklabels(methods, rotation=15, ha="right")
    axis.set_ylabel(metric.capitalize())
    axis.set_title(f"{metric.capitalize()} per income tier by method")
    axis.set_ylim(0, 1.05)
    axis.legend(title="Income tier", frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)

def plot_overall_metrics(results, out_path):
    metrics = ["accuracy", "precision", "recall", "f1"]
    methods = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(methods)

    figure, axis = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(methods):
        vals = [results[m][k] / (100 if k == "accuracy" else 1) for k in metrics]
        axis.bar(x + i * width, vals, width, label=m)

    axis.set_xticks(x + width * (len(methods) - 1) / 2)
    axis.set_xticklabels([k.upper() if k != "f1" else "F1" for k in metrics])
    axis.set_ylabel("Score (0-1)")
    axis.set_ylim(0, 1.05)
    axis.set_title("Overall performance by method")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
