import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_DIR = os.path.join(BASE_DIR, "simulation")
OUT_DIR = os.path.join(BASE_DIR, "Analysis", "dynamic_plots")

os.makedirs(OUT_DIR, exist_ok=True)

print("Project root:", BASE_DIR)
print("Simulation folder:", SIM_DIR)
print("Output folder:", OUT_DIR)

search_patterns = {
    "Video": [
        os.path.join(SIM_DIR, "train_nodes_250_video_mixed_workload_normal_seed42.csv"),
        os.path.join(SIM_DIR, "results_video", "train_nodes_*.csv"),
        os.path.join(BASE_DIR, "results_video", "train_nodes_*.csv"),
    ],
    "IoT": [
        os.path.join(SIM_DIR, "train_nodes_250_iot_mixed_workload_normal_seed42.csv"),
        os.path.join(SIM_DIR, "results_iot", "train_nodes_*.csv"),
        os.path.join(BASE_DIR, "results_iot", "train_nodes_*.csv"),
    ],
    "CI_CD": [
        os.path.join(SIM_DIR, "train_nodes_250_cicd_mixed_workload_normal_seed42.csv"),
        os.path.join(SIM_DIR, "results_cicd", "train_nodes_*.csv"),
        os.path.join(BASE_DIR, "results_cicd", "train_nodes_*.csv"),
    ],
}


def find_file(patterns):
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return files[0]
    return None


for label, patterns in search_patterns.items():
    path = find_file(patterns)

    if path is None:
        print(f"Missing result file for {label}")
        continue

    print(f"\nUsing {label} file:")
    print(path)

    df = pd.read_csv(path)

    required_columns = [
        "scenario_id",
        "workflow_time",
        "processing_time",
        "communication_delay",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        print(f"Skipping {label}. Missing columns: {missing}")
        print("Available columns:", list(df.columns))
        continue

    plt.figure(figsize=(10, 5))
    plt.plot(df["scenario_id"], df["workflow_time"], linewidth=1)
    plt.xlabel("Scenario ID")
    plt.ylabel("Workflow Time")
    plt.title(f"{label}: Dynamic Workflow Time Across Scenarios")
    plt.grid(True)
    plt.tight_layout()

    output_file_1 = os.path.join(OUT_DIR, f"dynamic_workflow_time_{label}.png")
    plt.savefig(output_file_1, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df["scenario_id"], df["processing_time"], linewidth=1, label="Processing Time")
    plt.plot(df["scenario_id"], df["communication_delay"], linewidth=1, label="Communication Delay")
    plt.plot(df["scenario_id"], df["workflow_time"], linewidth=1, label="Total Workflow Time")
    plt.xlabel("Scenario ID")
    plt.ylabel("Time")
    plt.title(f"{label}: Processing, Communication, and Total Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_file_2 = os.path.join(OUT_DIR, f"dynamic_breakdown_{label}.png")
    plt.savefig(output_file_2, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", output_file_1)
    print("Saved:", output_file_2)

print("\nDynamic scenario-wise plots completed.")