import os
import pandas as pd
import matplotlib.pyplot as plt


def load_training_file(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    if "workflow_time" not in df.columns:
        raise ValueError(
            f"'workflow_time' column not found in {path}. Columns: {list(df.columns)}"
        )

    return df


def summarize_workflow_time(df: pd.DataFrame) -> dict:
    workflow_time = df["workflow_time"]

    return {
        "count": int(workflow_time.count()),
        "average_time": float(workflow_time.mean()),
        "variation": float(workflow_time.std()),
        "minimum_time": float(workflow_time.min()),
        "lower_quartile": float(workflow_time.quantile(0.25)),
        "median_time": float(workflow_time.median()),
        "upper_quartile": float(workflow_time.quantile(0.75)),
        "maximum_time": float(workflow_time.max()),
    }


def main() -> None:
    base_folder = os.path.dirname(os.path.abspath(__file__))
    print("✅ Running infrastructure scaling analysis")
    print("📁 Working folder:", base_folder)

    node_counts = [15, 20, 30, 40, 50, 80, 100, 110, 120, 150, 250]

    input_files = {
        n: os.path.join(base_folder, f"train_nodes_{n}_baseline_normal_seed42.csv")
        for n in node_counts
    }

    summary_rows = []

    for number_of_nodes, file_path in input_files.items():
        df = load_training_file(file_path)
        stats = summarize_workflow_time(df)
        stats["number_of_nodes"] = number_of_nodes
        summary_rows.append(stats)

    summary = pd.DataFrame(summary_rows).sort_values("number_of_nodes").reset_index(drop=True)

    summary_file = os.path.join(base_folder, "sc1_summary.csv")
    summary.to_csv(summary_file, index=False)

    print("\n✅ Infrastructure scaling summary")
    print(
        summary[
            [
                "number_of_nodes",
                "count",
                "average_time",
                "variation",
                "minimum_time",
                "median_time",
                "maximum_time",
            ]
        ]
    )

    plt.figure(figsize=(8, 5))
    plt.plot(summary["number_of_nodes"], summary["average_time"], marker="o")
    plt.xlabel("Number of nodes")
    plt.ylabel("Average workflow time")
    plt.title("Average workflow time for fine-grained node scaling")
    plt.grid(True)
    plt.tight_layout()

    average_time_plot = os.path.join(base_folder, "sc1_nodes_vs_mean_workflow_time.png")
    plt.savefig(average_time_plot, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(summary["number_of_nodes"], summary["variation"], marker="o")
    plt.xlabel("Number of nodes")
    plt.ylabel("Variation in workflow time")
    plt.title("Workflow time variation for fine-grained node scaling")
    plt.grid(True)
    plt.tight_layout()

    variation_plot = os.path.join(base_folder, "sc1_nodes_vs_std_workflow_time.png")
    plt.savefig(variation_plot, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(summary["number_of_nodes"], summary["median_time"], marker="o", label="Median")
    plt.fill_between(
        summary["number_of_nodes"],
        summary["minimum_time"],
        summary["maximum_time"],
        alpha=0.2,
        label="Min-Max Range"
    )
    plt.xlabel("Number of nodes")
    plt.ylabel("Workflow time")
    plt.title("Workflow time stability across node scaling")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    band_plot = os.path.join(base_folder, "sc1_nodes_vs_minmax_band.png")
    plt.savefig(band_plot, dpi=200, bbox_inches="tight")
    plt.close()

    print("\n📁 Saved files:")
    print(" -", summary_file)
    print(" -", average_time_plot)
    print(" -", variation_plot)
    print(" -", band_plot)
    print("\n✅ Analysis completed")


if __name__ == "__main__":
    main()