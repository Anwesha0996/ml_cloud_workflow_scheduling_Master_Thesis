import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# SETTINGS
# =========================================================
profile = "mixed_workload"
condition = "normal"
seeds = [42, 43, 44, 45, 46]
folder = "simulation"

# =========================================================
# LOAD DATA
# =========================================================
summary_rows = []
all_workflow_times = []
labels = []

for seed in seeds:
    filename = f"train_nodes_250_{profile}_{condition}_seed{seed}.csv"
    path = os.path.join(folder, filename)

    if not os.path.isfile(path):
        print(f"Missing file: {filename}")
        continue

    df = pd.read_csv(path)

    if "workflow_time" not in df.columns:
        raise ValueError(f"'workflow_time' column missing in {filename}")

    mean_time = df["workflow_time"].mean()
    std_time = df["workflow_time"].std()
    min_time = df["workflow_time"].min()
    max_time = df["workflow_time"].max()

    summary_rows.append({
        "seed": seed,
        "mean_workflow_time": mean_time,
        "std_workflow_time": std_time,
        "min_workflow_time": min_time,
        "max_workflow_time": max_time
    })

    all_workflow_times.append(df["workflow_time"].values)
    labels.append(f"seed{seed}")

if not summary_rows:
    raise FileNotFoundError("No seed files were found. Check filenames and paths.")

summary_df = pd.DataFrame(summary_rows)

# Save numeric summary
summary_csv = os.path.join(folder, f"statistical_summary_{profile}_{condition}.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"Saved summary CSV: {summary_csv}")
print(summary_df)

# =========================================================
# GRAPH 1: Mean workflow time per seed with error bars
# =========================================================
plt.figure(figsize=(8, 5))
plt.errorbar(
    summary_df["seed"],
    summary_df["mean_workflow_time"],
    yerr=summary_df["std_workflow_time"],
    fmt='o-',
    capsize=5
)
plt.xlabel("Seed")
plt.ylabel("Mean workflow time")
plt.title(f"Mean workflow time across repeated runs\n({profile}, {condition})")
plt.grid(True)
plt.tight_layout()

plot1 = os.path.join(folder, f"statistical_mean_errorbars_{profile}_{condition}.png")
plt.savefig(plot1, dpi=200, bbox_inches="tight")
plt.close()

# =========================================================
# GRAPH 2: Boxplot of workflow time distribution per seed
# =========================================================
plt.figure(figsize=(9, 5))
plt.boxplot(all_workflow_times, tick_labels=labels)
plt.xlabel("Run")
plt.ylabel("Workflow time")
plt.title(f"Workflow time distribution across repeated runs\n({profile}, {condition})")
plt.grid(True)
plt.tight_layout()

plot2 = os.path.join(folder, f"statistical_boxplot_{profile}_{condition}.png")
plt.savefig(plot2, dpi=200, bbox_inches="tight")
plt.close()

# =========================================================
# GRAPH 3: Overall mean ± std across all runs
# =========================================================
overall_mean = summary_df["mean_workflow_time"].mean()
overall_std = summary_df["mean_workflow_time"].std()

plt.figure(figsize=(6, 5))
plt.bar(["overall"], [overall_mean], yerr=[overall_std], capsize=8)
plt.ylabel("Mean workflow time")
plt.title(f"Overall stability summary\n({profile}, {condition})")
plt.tight_layout()

plot3 = os.path.join(folder, f"statistical_overall_summary_{profile}_{condition}.png")
plt.savefig(plot3, dpi=200, bbox_inches="tight")
plt.close()

# =========================================================
# GRAPH 4: Min-max band + mean line
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(summary_df["seed"], summary_df["mean_workflow_time"], marker='o', label="Mean")
plt.fill_between(
    summary_df["seed"],
    summary_df["min_workflow_time"],
    summary_df["max_workflow_time"],
    alpha=0.2,
    label="Min-Max Range"
)
plt.xlabel("Seed")
plt.ylabel("Workflow time")
plt.title(f"Workflow time stability with min-max band\n({profile}, {condition})")
plt.grid(True)
plt.legend()
plt.tight_layout()

plot4 = os.path.join(folder, f"statistical_minmax_band_{profile}_{condition}.png")
plt.savefig(plot4, dpi=200, bbox_inches="tight")
plt.close()

print("\nSaved plots:")
print(" -", plot1)
print(" -", plot2)
print(" -", plot3)
print(" -", plot4)

print("\nOverall summary:")
print(f"Overall mean of run means: {overall_mean:.3f}")
print(f"Std of run means: {overall_std:.3f}")