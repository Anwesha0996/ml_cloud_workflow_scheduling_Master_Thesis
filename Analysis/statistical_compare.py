import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
seeds = [42, 43, 44, 45, 46]

def load_stats(folder_name, label):
    rows = []
    for seed in seeds:
        pattern = os.path.join(BASE_DIR, folder_name, f"train_nodes_250_*_mixed_workload_normal_seed{seed}.csv")
        files = glob.glob(pattern)
        print(f"{label} | seed{seed} | found: {[os.path.basename(f) for f in files]}")
        if not files:
            continue
        df = pd.read_csv(files[0])
        rows.append({
            "seed": seed,
            "mean_time": df["workflow_time"].mean(),
            "label": label
        })
    return pd.DataFrame(rows)

video = load_stats("results_video", "Video")
iot = load_stats("results_iot", "IoT")
cicd = load_stats("results_cicd", "CI/CD")

plt.figure(figsize=(8, 5))

any_data = False
for df in [video, iot, cicd]:
    if not df.empty:
        any_data = True
        plt.plot(df["seed"], df["mean_time"], marker="o", label=df["label"].iloc[0])

if not any_data:
    raise ValueError("No statistical data found. Check seed43–46 files in results folders.")

plt.xlabel("Seed")
plt.ylabel("Average Workflow Time")
plt.title("Statistical Reliability Across Seeds")
plt.grid(True)
plt.legend()
plt.tight_layout()

out = os.path.join(BASE_DIR, "statistical_compare.png")
plt.savefig(out, dpi=200)
print(f"\nSaved: {out}")