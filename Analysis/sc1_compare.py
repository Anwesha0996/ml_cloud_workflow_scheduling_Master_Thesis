import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_sc1(folder_name, label):
    files = glob.glob(os.path.join(BASE_DIR, folder_name, "train_nodes_*_*_mixed_workload_normal_seed42.csv"))
    rows = []
    for f in files:
        name = os.path.basename(f)
        m = re.search(r"train_nodes_(\d+)_", name)
        if not m:
            continue
        n_nodes = int(m.group(1))
        df = pd.read_csv(f)
        rows.append({"nodes": n_nodes, "mean_time": df["workflow_time"].mean(), "label": label})
    return pd.DataFrame(rows)

video = load_sc1("results_video", "Video")
iot = load_sc1("results_iot", "IoT")
cicd = load_sc1("results_cicd", "CI/CD")

plt.figure(figsize=(8,5))
for df in [video, iot, cicd]:
    if not df.empty:
        df = df.sort_values("nodes")
        plt.plot(df["nodes"], df["mean_time"], marker="o", label=df["label"].iloc[0])

plt.xlabel("Number of Nodes")
plt.ylabel("Average Workflow Time")
plt.title("SC1: Node Scaling Comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "sc1_compare.png"), dpi=200)