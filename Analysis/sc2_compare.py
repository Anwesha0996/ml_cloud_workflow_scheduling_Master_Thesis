import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
profiles = ["light_workload", "heavy_workload", "data_intensive_workload", "mixed_workload"]

def load_sc2(folder_name, label):
    rows = []
    for profile in profiles:
        pattern = os.path.join(BASE_DIR, folder_name, f"train_nodes_250_*_{profile}_normal_seed42.csv")
        files = glob.glob(pattern)
        print(f"{label} | {profile} | found: {[os.path.basename(f) for f in files]}")
        if not files:
            continue
        df = pd.read_csv(files[0])
        rows.append({
            "profile": profile,
            "mean_time": df["workflow_time"].mean(),
            "label": label
        })
    return pd.DataFrame(rows)

video = load_sc2("results_video", "Video")
iot = load_sc2("results_iot", "IoT")
cicd = load_sc2("results_cicd", "CI/CD")

all_df = pd.concat([video, iot, cicd], ignore_index=True)

if all_df.empty:
    raise ValueError("No SC2 data found. Check results_video/results_iot/results_cicd files.")

pivot = all_df.pivot(index="profile", columns="label", values="mean_time")
pivot = pivot.reindex(profiles)
print("\nSC2 pivot table:\n", pivot)

ax = pivot.plot(kind="bar", figsize=(9, 5))
ax.set_xlabel("Workload Profile")
ax.set_ylabel("Average Workflow Time")
ax.set_title("SC2: Workload Comparison")
ax.grid(True)
plt.tight_layout()

out = os.path.join(BASE_DIR, "sc2_compare.png")
plt.savefig(out, dpi=200)
print(f"\nSaved: {out}")