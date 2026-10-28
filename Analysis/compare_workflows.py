import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def find_result_file(folder_name: str) -> str:
    folder_path = os.path.join(BASE_DIR, folder_name)
    pattern = os.path.join(folder_path, "train_nodes_250_*_mixed_workload_normal_seed42.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No baseline result CSV found in {folder_path}")
    return files[0]

video = pd.read_csv(find_result_file("results_video"))
iot = pd.read_csv(find_result_file("results_iot"))
cicd = pd.read_csv(find_result_file("results_cicd"))

means = [video["workflow_time"].mean(), iot["workflow_time"].mean(), cicd["workflow_time"].mean()]
stds = [video["workflow_time"].std(), iot["workflow_time"].std(), cicd["workflow_time"].std()]

plt.figure(figsize=(8,5))
plt.bar(["Video", "IoT", "CI/CD"], means, yerr=stds, capsize=5)
plt.xlabel("Workflow Type")
plt.ylabel("Average Workflow Time")
plt.title("Baseline Comparison of Three Models")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "workflow_comparison.png"), dpi=200)