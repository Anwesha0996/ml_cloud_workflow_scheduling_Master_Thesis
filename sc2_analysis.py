# simulation/sc2_analysis.py
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    outdir = "simulation"

    files = {
        "short": "train_nodes_250_short.csv",
        "long": "train_nodes_250_long.csv",
        "dataheavy": "train_nodes_250_dataheavy.csv",
        "mixed": "train_nodes_250_mixed.csv",
    }

    summary = []

    for name, fname in files.items():
        path = os.path.join(outdir, fname)
        if not os.path.isfile(path):
            print(f"❌ Missing {fname}")
            continue

        df = pd.read_csv(path)

        summary.append({
            "workflow": name,
            "mean_workflow_time": df["workflow_time"].mean(),
            "std_workflow_time": df["workflow_time"].std(),
            "min_workflow_time": df["workflow_time"].min(),
            "max_workflow_time": df["workflow_time"].max(),
        })

    summary_df = pd.DataFrame(summary)

    # Save summary table
    out_csv = os.path.join(outdir, "sc2_summary.csv")
    summary_df.to_csv(out_csv, index=False)

    print("✅ SC2 summary saved:", out_csv)
    print(summary_df)

    # Plot: mean workflow time
    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["workflow"], summary_df["mean_workflow_time"])
    plt.ylabel("Mean Workflow Time")
    plt.xlabel("Workflow Type")
    plt.title("SC2: Workflow Type vs Mean Workflow Time")
    plt.tight_layout()

    out_plot = os.path.join(outdir, "sc2_workflow_vs_mean_time.png")
    plt.savefig(out_plot)
    plt.close()

    print("📊 Plot saved:", out_plot)


if __name__ == "__main__":
    main()
