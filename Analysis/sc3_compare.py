import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
conditions = ["fast_network", "slow_network", "light_memory_use", "heavy_memory_use"]

def load_sc3(folder_name, label):
    rows = []
    for cond in conditions:
        pattern = os.path.join(BASE_DIR, folder_name, f"train_nodes_250_*_mixed_workload_{cond}_seed42.csv")
        files = glob.glob(pattern)
        if not files:
            continue
        df = pd.read_csv(files[0])
        rows.append({"condition": cond, "mean_time": df["workflow_time"].mean(), "label": label})
    return pd.DataFrame(rows)

video = load_sc3("results_video", "Video")
iot = load_sc3("results_iot", "IoT")
cicd = load_sc3("results_cicd", "CI/CD")

all_df = pd.concat([video, iot, cicd], ignore_index=True)
pivot = all_df.pivot(index="condition", columns="label", values="mean_time")
pivot = pivot.loc[[c for c in conditions if c in pivot.index]]

pivot.plot(kind="bar", figsize=(9,5))
plt.xlabel("Environment Condition")
plt.ylabel("Average Workflow Time")
plt.title("SC3: Environment Comparison")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "sc3_compare.png"), dpi=200)