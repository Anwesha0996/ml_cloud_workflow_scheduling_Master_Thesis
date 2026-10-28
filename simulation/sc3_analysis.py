import os
import pandas as pd
import matplotlib.pyplot as plt


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    required = [
        "workflow_time",
        "processing_time",
        "communication_delay",
        "memory_use",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{os.path.basename(path)} missing columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    return df


def summarize(df: pd.DataFrame, label: str) -> dict:
    return {
        "condition": label,
        "n_rows": int(len(df)),
        "mean_workflow_time": float(df["workflow_time"].mean()),
        "std_workflow_time": float(df["workflow_time"].std()),
        "min_workflow_time": float(df["workflow_time"].min()),
        "max_workflow_time": float(df["workflow_time"].max()),
        "mean_processing_time": float(df["processing_time"].mean()),
        "mean_communication_delay": float(df["communication_delay"].mean()),
        "mean_memory_use": float(df["memory_use"].mean()),
        "corr_time_processing": float(df["workflow_time"].corr(df["processing_time"])),
        "corr_time_communication": float(df["workflow_time"].corr(df["communication_delay"])),
        "corr_time_memory": float(df["workflow_time"].corr(df["memory_use"])),
    }


def pretty_condition_name(name: str) -> str:
    mapping = {
        "fast_network": "Fast Network",
        "slow_network": "Slow Network",
        "light_memory_use": "Light Memory Use",
        "heavy_memory_use": "Heavy Memory Use",
    }
    return mapping.get(name, name.replace("_", " ").title())


def main():
    outdir = "simulation"

    profile = os.environ.get("WORKFLOW_PROFILE", "mixed_workload").strip().lower()

    files = {
        "fast_network": f"train_nodes_250_{profile}_fast_network.csv",
        "slow_network": f"train_nodes_250_{profile}_slow_network.csv",
        "light_memory_use": f"train_nodes_250_{profile}_light_memory_use.csv",
        "heavy_memory_use": f"train_nodes_250_{profile}_heavy_memory_use.csv",
    }

    summaries = []
    data = {}

    for cond, fname in files.items():
        path = os.path.join(outdir, fname)
        df = load_csv(path)
        data[cond] = df
        summaries.append(summarize(df, cond))

    summary_df = pd.DataFrame(summaries).sort_values("condition").reset_index(drop=True)
    summary_df["condition_label"] = summary_df["condition"].apply(pretty_condition_name)

    out_csv = os.path.join(outdir, f"sc3_summary_{profile}.csv")
    summary_df.to_csv(out_csv, index=False)

    print("✅ SC3 summary saved:", out_csv)
    print(
        summary_df[
            [
                "condition",
                "mean_workflow_time",
                "std_workflow_time",
                "mean_processing_time",
                "mean_communication_delay",
                "mean_memory_use",
                "corr_time_communication",
                "corr_time_memory",
            ]
        ]
    )

    # Plot 1: condition vs mean workflow time
    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["condition_label"], summary_df["mean_workflow_time"])
    plt.ylabel("Average Workflow Execution Time")
    plt.xlabel("System Condition")
    plt.title(f"Average Workflow Execution Time by Condition ({profile})")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    out_png1 = os.path.join(outdir, f"sc3_mean_time_{profile}.png")
    plt.savefig(out_png1, dpi=200, bbox_inches="tight")
    plt.close()

    # Plot 2: workflow time vs communication delay
    plt.figure(figsize=(8, 5))
    for cond, df in data.items():
        plt.scatter(
            df["communication_delay"],
            df["workflow_time"],
            label=pretty_condition_name(cond),
            s=10
        )
    plt.xlabel("Communication Delay")
    plt.ylabel("Workflow Execution Time")
    plt.title(f"Workflow Execution Time vs Communication Delay ({profile})")
    plt.legend()
    plt.tight_layout()

    out_png2 = os.path.join(outdir, f"sc3_time_vs_communication_{profile}.png")
    plt.savefig(out_png2, dpi=200, bbox_inches="tight")
    plt.close()

    # Plot 3: workflow time vs memory use
    plt.figure(figsize=(8, 5))
    for cond, df in data.items():
        plt.scatter(
            df["memory_use"],
            df["workflow_time"],
            label=pretty_condition_name(cond),
            s=10
        )
    plt.xlabel("Memory Usage")
    plt.ylabel("Workflow Execution Time")
    plt.title(f"Workflow Execution Time vs Memory Usage ({profile})")
    plt.legend()
    plt.tight_layout()

    out_png3 = os.path.join(outdir, f"sc3_time_vs_memory_{profile}.png")
    plt.savefig(out_png3, dpi=200, bbox_inches="tight")
    plt.close()

    print("\n📁 Saved plots:")
    print(" -", out_png1)
    print(" -", out_png2)
    print(" -", out_png3)
    print("\nDone ✅")


if __name__ == "__main__":
    main()