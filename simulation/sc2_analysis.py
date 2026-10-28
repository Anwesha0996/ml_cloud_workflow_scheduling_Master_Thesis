# simulation/sc2_analysis.py
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    output_folder = "simulation"

    # Updated workflow names (must match workflow_profiles.py)
    input_files = {
    "light_workload": "train_nodes_250_light_workload_normal.csv",
    "heavy_workload": "train_nodes_250_heavy_workload_normal.csv",
    "data_intensive_workload": "train_nodes_250_data_intensive_workload_normal.csv",
    "mixed_workload": "train_nodes_250_mixed_workload_normal.csv",
}

    summary_rows = []

    for workflow_name, filename in input_files.items():
        file_path = os.path.join(output_folder, filename)

        if not os.path.isfile(file_path):
            print(f"❌ Missing file: {filename}")
            continue

        df = pd.read_csv(file_path)

        summary_rows.append({
            "workflow_type": workflow_name,
            "average_time": df["workflow_time"].mean(),
            "variation": df["workflow_time"].std(),
            "minimum_time": df["workflow_time"].min(),
            "maximum_time": df["workflow_time"].max(),
        })

    summary_df = pd.DataFrame(summary_rows)

    # Save summary
    output_csv = os.path.join(output_folder, "sc2_summary.csv")
    summary_df.to_csv(output_csv, index=False)

    print("✅ SC2 summary saved:", output_csv)
    print(summary_df)

    # Plot: workflow type vs average time
    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["workflow_type"], summary_df["average_time"])

    plt.xlabel("Workflow type")
    plt.ylabel("Average workflow time")
    plt.title("Workflow type vs average workflow time")
    plt.xticks(rotation=20)
    plt.tight_layout()

    output_plot = os.path.join(output_folder, "sc2_workflow_vs_average_time.png")
    plt.savefig(output_plot)
    plt.close()

    print("📊 Plot saved:", output_plot)


if __name__ == "__main__":
    main()