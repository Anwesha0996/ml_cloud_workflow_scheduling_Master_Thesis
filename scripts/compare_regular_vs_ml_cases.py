#!/usr/bin/env python3
"""Create representative case-level regular-vs-ML simulation comparisons."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "case_comparison"

INPUT_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "scheduler_comparison" / "comparison_results.csv",
    PROJECT_ROOT / "outputs" / "ml_comparison" / "comparison_results.csv",
    PROJECT_ROOT / "comparison_results.csv",
    # Compatibility with the current simulation comparison script.
    PROJECT_ROOT
    / "outputs"
    / "ml_scheduler_comparison"
    / "comparison_results.csv",
]

REQUIRED_COLUMNS = [
    "scenario_id",
    "workflow_type",
    "workload_type",
    "environment_condition",
    "node_count",
    "regular_actual_time",
    "ml_predicted_time_for_selected_candidate",
    "ml_selected_actual_time",
    "absolute_improvement",
    "percent_improvement",
    "winner",
    "number_of_candidates",
]

WORKFLOWS = ("video", "iot", "cicd")
TOLERANCE = 1e-9


def find_input_file() -> Path:
    for path in INPUT_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "comparison_results.csv not found. First run the regular-vs-ML "
        "simulation comparison script."
    )


def recalculate_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Recalculate all final comparison fields from simulated actual times."""
    result = frame.copy()
    numeric_columns = [
        "regular_actual_time",
        "ml_predicted_time_for_selected_candidate",
        "ml_selected_actual_time",
        "node_count",
        "number_of_candidates",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    required_actual = result[
        ["regular_actual_time", "ml_selected_actual_time"]
    ]
    invalid_rows = required_actual.isna().any(axis=1)
    if invalid_rows.any():
        raise ValueError(
            f"{int(invalid_rows.sum())} rows have missing or non-numeric actual "
            "workflow times."
        )
    if np.isclose(result["regular_actual_time"], 0.0, atol=TOLERANCE).any():
        raise ValueError(
            "Cannot calculate percent improvement because regular_actual_time "
            "contains zero or near-zero values."
        )

    result["absolute_improvement"] = (
        result["regular_actual_time"] - result["ml_selected_actual_time"]
    )
    result["percent_improvement"] = (
        result["absolute_improvement"] / result["regular_actual_time"] * 100.0
    )

    ties = np.isclose(
        result["regular_actual_time"],
        result["ml_selected_actual_time"],
        rtol=0.0,
        atol=TOLERANCE,
    )
    result["winner"] = np.where(
        ties,
        "tie",
        np.where(
            result["ml_selected_actual_time"] < result["regular_actual_time"],
            "ml",
            "regular",
        ),
    )
    return result


def closest_row(
    frame: pd.DataFrame,
    target: float,
) -> pd.Series:
    """Choose a deterministic row nearest a target percent improvement."""
    ordered = frame.assign(
        _distance=(frame["percent_improvement"] - target).abs()
    ).sort_values(
        ["_distance", "scenario_id"],
        kind="stable",
    )
    return ordered.iloc[0].drop(labels="_distance")


def choose_representative_cases(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for workflow in WORKFLOWS:
        workflow_rows = frame[
            frame["workflow_type"].astype(str).str.lower().eq(workflow)
        ]
        if workflow_rows.empty:
            continue
        median_improvement = float(
            workflow_rows["percent_improvement"].median()
        )
        row = closest_row(workflow_rows, median_improvement).to_dict()
        row["workflow_median_percent_improvement"] = median_improvement
        records.append(row)
    columns = REQUIRED_COLUMNS + ["workflow_median_percent_improvement"]
    return pd.DataFrame(records, columns=columns)


def choose_extra_cases(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for workflow in WORKFLOWS:
        workflow_rows = frame[
            frame["workflow_type"].astype(str).str.lower().eq(workflow)
        ]
        if workflow_rows.empty:
            continue

        selections = [
            (
                "best_ml",
                workflow_rows.sort_values(
                    ["percent_improvement", "scenario_id"],
                    ascending=[False, True],
                    kind="stable",
                ).iloc[0],
            ),
            (
                "worst_ml",
                workflow_rows.sort_values(
                    ["percent_improvement", "scenario_id"],
                    ascending=[True, True],
                    kind="stable",
                ).iloc[0],
            ),
            ("tie_or_nearest_zero", closest_row(workflow_rows, 0.0)),
        ]
        for case_type, row in selections:
            record = row.to_dict()
            record["case_type"] = case_type
            records.append(record)
    columns = ["case_type"] + REQUIRED_COLUMNS
    return pd.DataFrame(records, columns=columns)


def grouped_summary(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    return (
        frame.groupby(group_column, dropna=False)
        .agg(
            scenario_count=("scenario_id", "count"),
            mean_regular_actual_time=("regular_actual_time", "mean"),
            mean_ml_selected_actual_time=("ml_selected_actual_time", "mean"),
            mean_absolute_improvement=("absolute_improvement", "mean"),
            mean_percent_improvement=("percent_improvement", "mean"),
        )
        .reset_index()
    )


def markdown_table(frame: pd.DataFrame) -> list[str]:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(columns))) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def case_interpretation(percent_improvement: float) -> str:
    if np.isclose(percent_improvement, 0.0, rtol=0.0, atol=TOLERANCE):
        return "Both produced equal simulated workflow time."
    if percent_improvement > 0:
        return "ML selected a better simulated placement."
    return "The regular simulated scheduler was better."


def display_workflow(workflow: str) -> str:
    return {"video": "Video", "iot": "IoT", "cicd": "CI/CD"}.get(
        workflow, workflow
    )


def save_plot(representatives: pd.DataFrame) -> None:
    if representatives.empty:
        return
    ordered = representatives.set_index("workflow_type").reindex(
        [workflow for workflow in WORKFLOWS if workflow in set(representatives["workflow_type"])]
    )
    x = np.arange(len(ordered))
    width = 0.36

    fig, axis = plt.subplots(figsize=(9, 6))
    axis.bar(
        x - width / 2,
        ordered["regular_actual_time"],
        width,
        label="Regular actual time",
        color="#777777",
    )
    axis.bar(
        x + width / 2,
        ordered["ml_selected_actual_time"],
        width,
        label="ML-selected actual time",
        color="#4472C4",
    )
    axis.set_xticks(x)
    axis.set_xticklabels([display_workflow(value) for value in ordered.index])
    axis.set_ylabel("Simulated workflow time")
    axis.set_title("Representative Regular vs ML-Selected Cases")
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "representative_cases_barplot.png", dpi=180)
    plt.close(fig)


def write_report(
    source_path: Path,
    frame: pd.DataFrame,
    representatives: pd.DataFrame,
    extras: pd.DataFrame,
) -> None:
    workflow_summary = grouped_summary(frame, "workflow_type")
    workload_summary = grouped_summary(frame, "workload_type")
    node_summary = grouped_summary(frame, "node_count")
    win_counts = (
        frame["winner"]
        .value_counts()
        .reindex(["ml", "regular", "tie"], fill_value=0)
        .rename_axis("winner")
        .reset_index(name="count")
    )

    lines = [
        "# Regular vs ML Case-Level Comparison",
        "",
        "This is a simulation-based comparison, not a real Kubernetes cluster result.",
        "",
        f"- Input: `{source_path.relative_to(PROJECT_ROOT)}`",
        f"- Total cases: {len(frame)}",
        "",
        "> **Warning:** `ml_predicted_time_for_selected_candidate` is not the "
        "final result. It is only the value used by the ML model to select a "
        "candidate. The final comparison must use `ml_selected_actual_time`.",
        "",
        "## Representative Cases",
        "",
        "Each representative is the case closest to its workflow's median percent "
        "improvement, avoiding selection of an unusually good or bad case.",
        "",
    ]

    for row in representatives.itertuples(index=False):
        lines.extend(
            [
                f"### {display_workflow(row.workflow_type)}",
                "",
                f"- Scenario: `{row.scenario_id}`",
                f"- Workflow type: `{row.workflow_type}`",
                f"- Workload type: `{row.workload_type}`",
                f"- Environment condition: `{row.environment_condition}`",
                f"- Node count: `{row.node_count}`",
                f"- Regular actual time: `{row.regular_actual_time:.6f}`",
                f"- ML predicted time for selected candidate: "
                f"`{row.ml_predicted_time_for_selected_candidate:.6f}`",
                f"- ML-selected actual time: "
                f"`{row.ml_selected_actual_time:.6f}`",
                f"- Absolute improvement: `{row.absolute_improvement:.6f}`",
                f"- Percent improvement: `{row.percent_improvement:.6f}%`",
                f"- Winner: `{row.winner}`",
                f"- Number of candidates: `{row.number_of_candidates}`",
                f"- Workflow median percent improvement: "
                f"`{row.workflow_median_percent_improvement:.6f}%`",
                f"- Interpretation: "
                f"{case_interpretation(row.percent_improvement)}",
                "",
            ]
        )

    lines.extend(
        [
            "## Best, Worst, and Nearest-Zero Cases",
            "",
        ]
    )
    compact_extras = extras[
        [
            "case_type",
            "workflow_type",
            "scenario_id",
            "regular_actual_time",
            "ml_selected_actual_time",
            "percent_improvement",
            "winner",
        ]
    ]
    lines.extend(markdown_table(compact_extras))

    lines.extend(["", "## Mean Result by Workflow Type", ""])
    lines.extend(markdown_table(workflow_summary))
    lines.extend(["", "## Mean Result by Workload Type", ""])
    lines.extend(markdown_table(workload_summary))
    lines.extend(["", "## Mean Result by Node Count", ""])
    lines.extend(markdown_table(node_summary))
    lines.extend(["", "## Win/Tie/Loss Counts", ""])
    lines.extend(markdown_table(win_counts))
    lines.extend(
        [
            "",
            "## Run",
            "",
            "```bash",
            "python scripts/compare_regular_vs_ml_cases.py",
            "```",
            "",
        ]
    )
    (OUTPUT_DIR / "regular_vs_ml_case_comparison_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    source_path = find_input_file()
    frame = pd.read_csv(source_path, low_memory=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "comparison_results.csv is missing required columns: "
            + ", ".join(missing)
        )

    frame = recalculate_results(frame)
    representatives = choose_representative_cases(frame)
    extras = choose_extra_cases(frame)

    missing_workflows = [
        workflow
        for workflow in WORKFLOWS
        if workflow not in set(representatives["workflow_type"])
    ]
    if missing_workflows:
        raise ValueError(
            "No comparison cases found for workflows: "
            + ", ".join(missing_workflows)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    representatives.to_csv(
        OUTPUT_DIR / "representative_cases_by_workflow.csv", index=False
    )
    extras.to_csv(
        OUTPUT_DIR / "best_worst_tie_cases_by_workflow.csv", index=False
    )
    save_plot(representatives)
    write_report(source_path, frame, representatives, extras)

    print("Case-level comparison completed.")
    print(f"Input: {source_path.relative_to(PROJECT_ROOT)}")
    print(f"Representative workflows: {len(representatives)}")
    print(f"Extra cases: {len(extras)}")
    print(f"Outputs: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
