import json
import numpy as np
import matplotlib.pyplot as plt

with open("tab.json", "r") as f:
    metrics_data = json.load(f)["metrics"]

methods = ["EUC", "LOG", "AIRM", "AIRM-GMM"]

colors = {
    "EUC": "#332288",
    "LOG": "#117733",
    "AIRM": "#44AA99",
    "AIRM-GMM": "#DDCC77",
}

method_labels = {
    "EUC": "EUC (\u03B1=1.08)",
    "LOG": "LOG (\u03B1=1.06)",
    "AIRM": "AIRM (\u03B1=1.00)",
    "AIRM-GMM": "AIRM GMM",
}

metrics = ["IOU", "DICE", "PREC", "RECL"]
metric_labels = {
    "IOU": "IoU",
    "DICE": "Dice",
    "PREC": "Precision",
    "RECL": "Recall",
}

x = np.arange(len(metrics))
width = 0.2

fig, ax = plt.subplots(figsize=(8, 6))

for i, method in enumerate(methods):
    means = [metrics_data[method]["aggregate"][m]["mean"] for m in metrics]
    ax.bar(x + i * width, means, width,
           label=method_labels[method],
           color=colors[method])

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([metric_labels[m] for m in metrics])
ax.set_ylabel("Mean")
ax.set_xlabel("Metric")
ax.set_title("Metrics (K=3)")
ax.set_ylim(0.0, 1.0)
ax.legend()
fig.tight_layout()
fig.savefig("methods_4.png", dpi=300)
plt.close(fig)
