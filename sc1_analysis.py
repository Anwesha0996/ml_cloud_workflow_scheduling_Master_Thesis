# simulation/sc1_analysis.py
import os
import pandas as pd
import matplotlib.pyplot as plt


def load_train_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    if "workflow_time" not in df.columns:
        raise ValueError(f"'workflow_time' column not found in {path}. Columns: {list(df.columns)}")
    return df


def summarize_one(df: pd.DataFrame) -> dict:
    s = df["workflow_time"]
    return {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
    }


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("✅ Running SC1 analysis. Folder:", base_dir)

    files = {
        50:  os.path.join(base_dir, "train_nodes_50.csv"),
        100: os.path.join(base_dir, "train_nodes_100.csv"),
        150: os.path.join(base_dir, "train_nodes_150.csv"),
        250: os.path.join(base_dir, "train_nodes_250.csv"),
    }

    rows = []
    for nodes, path in files.items():
        df = load_train_csv(path)
        stats = summarize_one(df)
        stats["nodes"] = nodes
        rows.append(stats)

    summary = pd.DataFrame(rows).sort_values("nodes").reset_index(drop=True)

    # Save summary table
    out_csv = os.path.join(base_dir, "sc1_summary.csv")
    summary.to_csv(out_csv, index=False)

    print("\n✅ SC1 Summary (Node Scaling)")
    print(summary[["nodes", "count", "mean", "std", "min", "median", "max"]])

    # Plot 1: nodes vs mean workflow time
    plt.figure()
    plt.plot(summary["nodes"], summary["mean"], marker="o")
    plt.xlabel("Number of nodes")
    plt.ylabel("Mean workflow_time")
    plt.title("SC1: Node scaling vs mean workflow time")
    plt.grid(True)
    out_png1 = os.path.join(base_dir, "sc1_nodes_vs_mean_workflow_time.png")
    plt.savefig(out_png1, dpi=200, bbox_inches="tight")
    plt.close()

    # Plot 2: nodes vs variability (std)
    plt.figure()
    plt.plot(summary["nodes"], summary["std"], marker="o")
    plt.xlabel("Number of nodes")
    plt.ylabel("Std dev of workflow_time")
    plt.title("SC1: Node scaling vs workflow time variability")
    plt.grid(True)
    out_png2 = os.path.join(base_dir, "sc1_nodes_vs_std_workflow_time.png")
    plt.savefig(out_png2, dpi=200, bbox_inches="tight")
    plt.close()

    print("\n📁 Saved files:")
    print(" -", out_csv)
    print(" -", out_png1)
    print(" -", out_png2)
    print("\nDone ✅")


if __name__ == "__main__":
    main()
