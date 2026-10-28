import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_DIR = os.path.join(BASE_DIR, "simulation")

def find_file(patterns):
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return files[0]
    return None

# Find files
video_file = find_file([
    os.path.join(BASE_DIR, "results_video", "train_nodes_*.csv"),
    os.path.join(SIM_DIR, "results_video", "train_nodes_*.csv"),
])

iot_file = find_file([
    os.path.join(SIM_DIR, "train_nodes_250_iot_*.csv"),
])

cicd_file = find_file([
    os.path.join(SIM_DIR, "train_nodes_250_cicd_*.csv"),
])

print("Video:", video_file)
print("IoT:", iot_file)
print("CI/CD:", cicd_file)

# Load data
dfs = {}

if video_file:
    dfs["Video"] = pd.read_csv(video_file)

if iot_file:
    dfs["IoT"] = pd.read_csv(iot_file)

if cicd_file:
    dfs["CI/CD"] = pd.read_csv(cicd_file)

# Plot comparison
plt.figure(figsize=(10, 5))

for label, df in dfs.items():
    if "scenario_id" in df.columns and "workflow_time" in df.columns:
        plt.plot(
            df["scenario_id"],
            df["workflow_time"],
            linewidth=1,
            label=label
        )

plt.xlabel("Scenario ID")
plt.ylabel("Workflow Time")
plt.title("Dynamic Comparison of Workflows (Video vs IoT vs CI/CD)")
plt.grid(True)
plt.legend()
plt.tight_layout()

out = os.path.join(BASE_DIR, "Analysis", "dynamic_plots", "dynamic_compare_all.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", out)