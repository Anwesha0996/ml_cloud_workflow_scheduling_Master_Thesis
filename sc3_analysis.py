# simulation/sc3_analysis.py
import os
import pandas as pd
import matplotlib.pyplot as plt


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)

    required = ["workflow_time", "network_cost", "storage_cost", "compute_cost"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)} missing columns: {missing}. Found: {list(df.columns)}")
    return df


def summarize(df: pd.DataFrame, label: str) -> dict:
    return {
        "condition": label,
        "n_rows": int(len(df)),
        "mean_workflow_time": float(df["workflow_time"].mean()),
        "std_workflow_time": float(df["workflow_time"].std()),
        "mean_network_cost": float(df["network_cost"].mean()),
        "mean_storage_cost": float(df["storage_cost"].mean()),
        "mean_compute_cost": float(df["compute_cost"].mean()),
        "corr_time_network": float(df["workflow_time"].corr(df["network_cost"])),
        "corr_time_storage": float(df["workflow_time"].corr(df["storage_cost"])),
        "corr_time_compute": float(df["workflow_time"].corr(df["compute_cost"])),
    }


def main():
    outdir = "simulation"

    profile = os.environ.get("WORKFLOW_PROFILE", "baseline").strip().lower()

    files = {
        "network_good":  f"train_nodes_250_{profile}_network_good.csv",
        "network_bad":   f"train_nodes_250_{profile}_network_bad.csv",
        "storage_light": f"train_nodes_250_{profile}_storage_light.csv",
        "storage_heavy": f"train_nodes_250_{profile}_storage_heavy.csv",
    }

    # Load + summarize
    summaries = []
    data = {}
    for cond, fname in files.items():
        path = os.path.join(outdir, fname)
        df = load_csv(path)
        data[cond] = df
        summaries.append(summarize(df, cond))

    summary_df = pd.DataFrame(summaries).sort_values("condition").reset_index(drop=True)

    # Save summary table
    out_csv = os.path.join(outdir, f"sc3_summary_{profile}.csv")
    summary_df.to_csv(out_csv, index=False)

    print("✅ SC3 summary saved:", out_csv)
    print(summary_df[[
        "condition",
        "mean_workflow_time",
        "mean_network_cost",
        "mean_storage_cost",
        "corr_time_network",
        "corr_time_storage"
    ]])

    # Plot 1: condition vs mean workflow_time
    plt.figure()
    plt.bar(summary_df["condition"], summary_df["mean_workflow_time"])
    plt.ylabel("Mean workflow_time")
    plt.xlabel("SC3 condition")
    plt.title(f"SC3: Mean workflow_time by condition ({profile})")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out_png1 = os.path.join(outdir, f"sc3_mean_time_{profile}.png")
    plt.savefig(out_png1, dpi=200, bbox_inches="tight")
    plt.close()

    # Plot 2: workflow_time vs network_cost (scatter per condition)
    plt.figure()
    for cond, df in data.items():
        plt.scatter(df["network_cost"], df["workflow_time"], label=cond, s=10)
    plt.xlabel("network_cost")
    plt.ylabel("workflow_time")
    plt.title(f"SC3: workflow_time vs network_cost ({profile})")
    plt.legend()
    plt.tight_layout()
    out_png2 = os.path.join(outdir, f"sc3_time_vs_network_{profile}.png")
    plt.savefig(out_png2, dpi=200, bbox_inches="tight")
    plt.close()

    # Plot 3: workflow_time vs storage_cost (scatter per condition)
    plt.figure()
    for cond, df in data.items():
        plt.scatter(df["storage_cost"], df["workflow_time"], label=cond, s=10)
    plt.xlabel("storage_cost")
    plt.ylabel("workflow_time")
    plt.title(f"SC3: workflow_time vs storage_cost ({profile})")
    plt.legend()
    plt.tight_layout()
    out_png3 = os.path.join(outdir, f"sc3_time_vs_storage_{profile}.png")
    plt.savefig(out_png3, dpi=200, bbox_inches="tight")
    plt.close()

    print("\n📁 Saved plots:")
    print(" -", out_png1)
    print(" -", out_png2)
    print(" -", out_png3)
    print("\nDone ✅")


if __name__ == "__main__":
    main()
