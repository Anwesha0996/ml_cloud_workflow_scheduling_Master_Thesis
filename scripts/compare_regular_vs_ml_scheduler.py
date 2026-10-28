#!/usr/bin/env python3
"""Compare regular scheduling with ML-selected scheduling fairly."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANDOM_STATE = 42
SAFE_FEATURES = [
    "workflow_type",
    "workload_type",
    "environment_condition",
    "node_count",
    "communication_scale",
    "memory_scale",
    "num_tasks",
    "num_edges",
    "dag_depth",
    "distinct_nodes_used",
    "dag_crossings",
    "colocated_edges",
]
RESULT_COLUMNS = [
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

WORKLOAD_PROFILES = {
    "light": {"runtime_scale": 0.7, "comm_scale": 1.0, "memory_scale": 1.0},
    "heavy": {"runtime_scale": 1.5, "comm_scale": 1.0, "memory_scale": 1.0},
    "data_intensive": {
        "runtime_scale": 1.1,
        "comm_scale": 1.6,
        "memory_scale": 2.0,
    },
    "mixed": {"runtime_scale": 1.2, "comm_scale": 1.3, "memory_scale": 1.3},
}
ENVIRONMENT_CONDITIONS = {
    "normal": {"comm_scale": 1.0, "memory_scale": 1.0},
    "fast_network": {"comm_scale": 0.6, "memory_scale": 1.0},
    "slow_network": {"comm_scale": 2.0, "memory_scale": 1.0},
    "light_memory_use": {"comm_scale": 1.0, "memory_scale": 0.7},
    "heavy_memory_use": {"comm_scale": 1.0, "memory_scale": 2.0},
}
REGULAR_LABELS = {"regular", "default", "baseline", "random"}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Compare regular actual workflow time with ML-selected actual workflow "
            "time using either the repository simulator or strict offline re-ranking."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=project_root
        / "outputs"
        / "rf_regression"
        / "rf_regression_pipeline.joblib",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=project_root
        / "outputs"
        / "feature_audit"
        / "extracted_ml_dataset.csv",
    )
    parser.add_argument(
        "--simulation-dir",
        type=Path,
        default=project_root / "simulation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "ml_scheduler_comparison",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "simulation", "offline"],
        default="auto",
    )
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--scenarios-per-configuration", type=int, default=1)
    parser.add_argument(
        "--workflows",
        nargs="+",
        choices=["video", "iot", "cicd"],
        default=["video", "iot", "cicd"],
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=list(WORKLOAD_PROFILES),
        default=list(WORKLOAD_PROFILES),
    )
    parser.add_argument(
        "--environments",
        nargs="+",
        choices=list(ENVIRONMENT_CONDITIONS),
        default=list(ENVIRONMENT_CONDITIONS),
    )
    parser.add_argument(
        "--node-counts",
        nargs="+",
        type=int,
        default=[15, 30, 60, 120, 250],
    )
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def load_model(path: Path) -> tuple[Any, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing trained model: {path}")
    artifact = joblib.load(path)
    if isinstance(artifact, dict) and "pipeline" in artifact:
        pipeline = artifact["pipeline"]
        features = list(artifact.get("features", SAFE_FEATURES))
    else:
        pipeline = artifact
        features = SAFE_FEATURES
    if features != SAFE_FEATURES:
        raise ValueError(
            "The model feature list does not match the required 12 safe features. "
            f"Model features: {features}"
        )
    return pipeline, features


def required_simulation_files(simulation_dir: Path, workflows: list[str]) -> list[Path]:
    files = [simulation_dir / "nodes.csv", simulation_dir / "timeseries.csv"]
    for workflow in workflows:
        files.extend(
            [
                simulation_dir / f"application_components_{workflow}.csv",
                simulation_dir / f"application_dag_{workflow}.csv",
            ]
        )
    return files


def simulator_available(simulation_dir: Path, workflows: list[str]) -> bool:
    return all(path.is_file() for path in required_simulation_files(simulation_dir, workflows))


def longest_path_depth(edges: pd.DataFrame) -> int:
    nodes = set(edges["src"]).union(edges["dst"])
    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree = {node: 0 for node in nodes}
    for source, target in edges[["src", "dst"]].itertuples(index=False, name=None):
        adjacency[source].append(target)
        indegree[target] += 1
    queue = deque(node for node, degree in indegree.items() if degree == 0)
    distance = {node: 0 for node in nodes}
    visited = 0
    while queue:
        source = queue.popleft()
        visited += 1
        for target in adjacency[source]:
            distance[target] = max(distance[target], distance[source] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != len(nodes):
        raise ValueError("Application graph contains a cycle.")
    return max(distance.values(), default=0)


def calculate_node_condition_score(row: pd.Series) -> float:
    """Match simulation/training_data_generator.py exactly."""
    cpu_usage = float(row["cpu"]) / 100.0
    memory_fraction = float(row["memory"]) / max(
        float(row.get("memory_capacity", 16000.0)), 1.0
    )
    resource_pressure = 1.0 + 1.8 * cpu_usage + 1.2 * memory_fraction
    network_pressure = (
        1.0
        + float(row["latency"]) / 120.0
        + 200.0 / max(float(row["bandwidth"]), 1.0)
        + float(row["packet_loss"]) / 4.0
    )
    reliability_pressure = 1.0 + (1.0 - float(row["availability"])) * 8.0
    return resource_pressure * network_pressure * reliability_pressure


def build_node_condition_scores(
    nodes: pd.DataFrame,
    timeseries: pd.DataFrame,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    snapshots = []
    for node_id in nodes["node_id"]:
        matching = timeseries[timeseries["node_id"] == node_id]
        if not matching.empty:
            snapshots.append(matching.iloc[[int(rng.integers(0, len(matching)))]])
    if not snapshots:
        raise RuntimeError("No node snapshots are available for this node subset.")
    snapshot_frame = pd.concat(snapshots, ignore_index=True).merge(
        nodes, on="node_id", how="left", suffixes=("", "_node")
    )
    scores = {
        str(row["node_id"]): calculate_node_condition_score(row)
        for _, row in snapshot_frame.iterrows()
    }
    average_score = float(np.mean(list(scores.values())))
    for node_id in nodes["node_id"]:
        scores.setdefault(str(node_id), average_score)
    return scores


def calculate_workflow_time(
    node_scores: dict[str, float],
    components: pd.DataFrame,
    dag: pd.DataFrame,
    assignment: dict[str, str],
    runtime_scale: float,
    communication_scale: float,
    memory_scale: float,
) -> float:
    processing_time = 0.0
    for component in components.itertuples(index=False):
        node_id = assignment[component.component]
        base_time = (
            0.7 * float(component.cpu_request)
            + 0.00035 * memory_scale * float(component.memory_request)
        )
        processing_time += runtime_scale * base_time * node_scores[node_id]

    communication_delay = 0.0
    for source, target in dag[["src", "dst"]].itertuples(index=False, name=None):
        source_node = assignment[source]
        target_node = assignment[target]
        if source_node != target_node:
            communication_delay += communication_scale * 0.25 * (
                node_scores[source_node] + node_scores[target_node]
            )
    return float(processing_time + communication_delay)


def assignment_feature_row(
    assignment: dict[str, str],
    dag: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    crossings = 0
    colocated = 0
    for source, target in dag[["src", "dst"]].itertuples(index=False, name=None):
        if assignment[source] == assignment[target]:
            colocated += 1
        else:
            crossings += 1
    return {
        **metadata,
        "distinct_nodes_used": len(set(assignment.values())),
        "dag_crossings": crossings,
        "colocated_edges": colocated,
    }


def random_assignment(
    components: list[str],
    node_ids: list[str],
    rng: np.random.Generator,
) -> dict[str, str]:
    return {component: str(rng.choice(node_ids)) for component in components}


def winner_label(regular_time: float, ml_time: float) -> str:
    if np.isclose(regular_time, ml_time, rtol=1e-9, atol=1e-9):
        return "tie"
    return "ml" if ml_time < regular_time else "regular"


def run_simulation_comparison(
    pipeline: Any,
    simulation_dir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    nodes_all = pd.read_csv(simulation_dir / "nodes.csv")
    timeseries = pd.read_csv(simulation_dir / "timeseries.csv")
    nodes_all["node_id"] = nodes_all["node_id"].astype(str)
    timeseries["node_id"] = timeseries["node_id"].astype(str)

    if max(args.node_counts) > len(nodes_all):
        raise ValueError(
            f"Requested {max(args.node_counts)} nodes, but nodes.csv has "
            f"{len(nodes_all)}."
        )
    if args.candidates < 2:
        raise ValueError("--candidates must be at least 2.")
    if args.scenarios_per_configuration < 1:
        raise ValueError("--scenarios-per-configuration must be at least 1.")

    workflow_data = {}
    for workflow in args.workflows:
        components = pd.read_csv(
            simulation_dir / f"application_components_{workflow}.csv"
        )
        dag = pd.read_csv(simulation_dir / f"application_dag_{workflow}.csv")
        workflow_data[workflow] = {
            "components": components,
            "dag": dag,
            "num_tasks": len(
                set(dag["src"].astype(str)).union(dag["dst"].astype(str))
            ),
            "num_edges": len(dag),
            "dag_depth": longest_path_depth(dag),
        }

    results = []
    scenario_number = 0
    for workflow in args.workflows:
        workflow_info = workflow_data[workflow]
        components_frame = workflow_info["components"]
        dag = workflow_info["dag"]
        component_names = components_frame["component"].astype(str).tolist()

        for workload in args.workloads:
            profile = WORKLOAD_PROFILES[workload]
            for environment in args.environments:
                condition = ENVIRONMENT_CONDITIONS[environment]
                communication_scale = (
                    profile["comm_scale"] * condition["comm_scale"]
                )
                memory_scale = (
                    profile["memory_scale"] * condition["memory_scale"]
                )
                for node_count in args.node_counts:
                    nodes = nodes_all.iloc[:node_count].copy()
                    node_ids = nodes["node_id"].tolist()
                    for replicate in range(args.scenarios_per_configuration):
                        scenario_seed = args.random_state + scenario_number
                        rng = np.random.default_rng(scenario_seed)
                        node_scores = build_node_condition_scores(
                            nodes, timeseries, seed=scenario_seed
                        )

                        # Candidate zero is the existing default random scheduler.
                        regular_assignment = random_assignment(
                            component_names, node_ids, rng
                        )
                        assignments = [regular_assignment]
                        assignments.extend(
                            random_assignment(component_names, node_ids, rng)
                            for _ in range(args.candidates - 1)
                        )

                        metadata = {
                            "workflow_type": workflow,
                            "workload_type": workload,
                            "environment_condition": environment,
                            "node_count": node_count,
                            "communication_scale": communication_scale,
                            "memory_scale": memory_scale,
                            "num_tasks": workflow_info["num_tasks"],
                            "num_edges": workflow_info["num_edges"],
                            "dag_depth": workflow_info["dag_depth"],
                        }
                        candidate_features = pd.DataFrame(
                            [
                                assignment_feature_row(assignment, dag, metadata)
                                for assignment in assignments
                            ]
                        )[SAFE_FEATURES]
                        predictions = pipeline.predict(candidate_features)
                        selected_index = int(np.argmin(predictions))
                        selected_assignment = assignments[selected_index]

                        regular_actual = calculate_workflow_time(
                            node_scores,
                            components_frame,
                            dag,
                            regular_assignment,
                            profile["runtime_scale"],
                            communication_scale,
                            memory_scale,
                        )
                        ml_actual = calculate_workflow_time(
                            node_scores,
                            components_frame,
                            dag,
                            selected_assignment,
                            profile["runtime_scale"],
                            communication_scale,
                            memory_scale,
                        )
                        absolute_improvement = regular_actual - ml_actual
                        percent_improvement = (
                            absolute_improvement / regular_actual * 100.0
                        )
                        scenario_id = (
                            f"{workflow}-{workload}-{environment}-nodes{node_count}"
                            f"-rep{replicate}"
                        )
                        results.append(
                            {
                                "scenario_id": scenario_id,
                                "workflow_type": workflow,
                                "workload_type": workload,
                                "environment_condition": environment,
                                "node_count": node_count,
                                "regular_actual_time": regular_actual,
                                "ml_predicted_time_for_selected_candidate": float(
                                    predictions[selected_index]
                                ),
                                "ml_selected_actual_time": ml_actual,
                                "absolute_improvement": absolute_improvement,
                                "percent_improvement": percent_improvement,
                                "winner": winner_label(regular_actual, ml_actual),
                                "number_of_candidates": len(assignments),
                            }
                        )
                        scenario_number += 1
    return pd.DataFrame(results, columns=RESULT_COLUMNS)


def run_offline_comparison(
    pipeline: Any,
    dataset_path: Path,
) -> pd.DataFrame | None:
    if not dataset_path.is_file():
        return None
    frame = pd.read_csv(dataset_path, low_memory=False)
    required = set(
        SAFE_FEATURES
        + [
            "source_file",
            "scenario_id",
            "scheduler_type",
            "candidate_id",
            "workflow_time",
        ]
    )
    if not required.issubset(frame.columns):
        return None

    usable = frame.dropna(
        subset=SAFE_FEATURES
        + ["source_file", "scenario_id", "scheduler_type", "candidate_id", "workflow_time"]
    ).copy()
    if usable.empty:
        return None
    usable["scheduler_type_normalized"] = (
        usable["scheduler_type"].astype(str).str.strip().str.lower()
    )

    results = []
    for (_, _), group in usable.groupby(["source_file", "scenario_id"], sort=False):
        regular = group[
            group["scheduler_type_normalized"].isin(REGULAR_LABELS)
        ]
        if regular.empty or group["candidate_id"].nunique() < 2:
            continue
        predictions = pipeline.predict(group[SAFE_FEATURES])
        selected_position = int(np.argmin(predictions))
        selected = group.iloc[selected_position]
        regular_row = regular.iloc[0]
        regular_actual = float(regular_row["workflow_time"])
        ml_actual = float(selected["workflow_time"])
        absolute_improvement = regular_actual - ml_actual
        results.append(
            {
                "scenario_id": selected["scenario_id"],
                "workflow_type": selected["workflow_type"],
                "workload_type": selected["workload_type"],
                "environment_condition": selected["environment_condition"],
                "node_count": selected["node_count"],
                "regular_actual_time": regular_actual,
                "ml_predicted_time_for_selected_candidate": float(
                    predictions[selected_position]
                ),
                "ml_selected_actual_time": ml_actual,
                "absolute_improvement": absolute_improvement,
                "percent_improvement": (
                    absolute_improvement / regular_actual * 100.0
                ),
                "winner": winner_label(regular_actual, ml_actual),
                "number_of_candidates": len(group),
            }
        )
    if not results:
        return None
    return pd.DataFrame(results, columns=RESULT_COLUMNS)


def grouped_summary(results: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    summaries = []
    for value, group in results.groupby(column, dropna=False):
        summaries.append(
            {
                column: None if pd.isna(value) else value,
                "scenario_count": int(len(group)),
                "mean_regular_actual_time": float(group["regular_actual_time"].mean()),
                "mean_ml_selected_actual_time": float(
                    group["ml_selected_actual_time"].mean()
                ),
                "mean_percent_improvement": float(
                    group["percent_improvement"].mean()
                ),
                "ml_win_count": int((group["winner"] == "ml").sum()),
                "regular_win_count": int((group["winner"] == "regular").sum()),
                "tie_count": int((group["winner"] == "tie").sum()),
            }
        )
    return summaries


def build_summary(results: pd.DataFrame, evaluation_type: str) -> dict[str, Any]:
    scenario_count = len(results)
    ml_wins = int((results["winner"] == "ml").sum())
    regular_wins = int((results["winner"] == "regular").sum())
    ties = int((results["winner"] == "tie").sum())
    return {
        "comparison_possible": True,
        "evaluation_type": evaluation_type,
        "random_state": RANDOM_STATE,
        "scenario_count": scenario_count,
        "mean_regular_actual_time": float(results["regular_actual_time"].mean()),
        "mean_ml_selected_actual_time": float(
            results["ml_selected_actual_time"].mean()
        ),
        "mean_percent_improvement": float(results["percent_improvement"].mean()),
        "median_percent_improvement": float(
            results["percent_improvement"].median()
        ),
        "ml_win_count": ml_wins,
        "regular_scheduler_win_count": regular_wins,
        "tie_count": ties,
        "ml_win_rate_percent": float(ml_wins / scenario_count * 100.0),
        "by_workload_type": grouped_summary(results, "workload_type"),
        "by_node_count": grouped_summary(results, "node_count"),
        "by_environment_condition": grouped_summary(
            results, "environment_condition"
        ),
    }


def save_plots(results: pd.DataFrame, output_dir: Path) -> None:
    means = [
        results["regular_actual_time"].mean(),
        results["ml_selected_actual_time"].mean(),
    ]
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.bar(["Regular scheduler", "ML-selected"], means, color=["#777777", "#4472C4"])
    axis.set_ylabel("Mean actual/simulated workflow time")
    axis.set_title("Regular vs ML-Selected Workflow Time")
    fig.tight_layout()
    fig.savefig(output_dir / "average_workflow_time.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(results["percent_improvement"], bins=30, color="#4472C4", alpha=0.85)
    axis.axvline(0, color="black", linestyle="--")
    axis.set_xlabel("Percent improvement")
    axis.set_ylabel("Scenario count")
    axis.set_title("ML Scheduling Improvement Distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "percent_improvement_distribution.png", dpi=180)
    plt.close(fig)

    workload = (
        results.groupby("workload_type")["percent_improvement"]
        .mean()
        .sort_values()
    )
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(workload.index, workload.values, color="#4472C4")
    axis.axhline(0, color="black", linewidth=1)
    axis.set_ylabel("Mean percent improvement")
    axis.set_title("Improvement by Workload Type")
    fig.tight_layout()
    fig.savefig(output_dir / "improvement_by_workload_type.png", dpi=180)
    plt.close(fig)

    node_count = (
        results.groupby("node_count")["percent_improvement"].mean().sort_index()
    )
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(node_count.index.astype(str), node_count.values, color="#4472C4")
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xlabel("Node count")
    axis.set_ylabel("Mean percent improvement")
    axis.set_title("Improvement by Node Count")
    fig.tight_layout()
    fig.savefig(output_dir / "improvement_by_node_count.png", dpi=180)
    plt.close(fig)

    regular = results["regular_actual_time"]
    ml_actual = results["ml_selected_actual_time"]
    lower = min(regular.min(), ml_actual.min())
    upper = max(regular.max(), ml_actual.max())
    fig, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(regular, ml_actual, alpha=0.55, s=20, edgecolors="none")
    axis.plot([lower, upper], [lower, upper], color="black", linestyle="--")
    axis.set_xlabel("Regular actual/simulated workflow time")
    axis.set_ylabel("ML-selected actual/simulated workflow time")
    axis.set_title("Regular vs ML-Selected Actual Time")
    fig.tight_layout()
    fig.savefig(output_dir / "regular_vs_ml_actual_scatter.png", dpi=180)
    plt.close(fig)


def markdown_table(records: list[dict[str, Any]], group_column: str) -> list[str]:
    lines = [
        f"| {group_column} | Scenarios | Regular mean | ML mean | "
        "Mean improvement % | ML wins | Regular wins | Ties |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record[group_column]} | {record['scenario_count']} | "
            f"{record['mean_regular_actual_time']:.4f} | "
            f"{record['mean_ml_selected_actual_time']:.4f} | "
            f"{record['mean_percent_improvement']:.4f} | "
            f"{record['ml_win_count']} | {record['regular_win_count']} | "
            f"{record['tie_count']} |"
        )
    return lines


def save_success_report(
    output_dir: Path,
    summary: dict[str, Any],
    evaluation_type: str,
    limitation: str,
) -> None:
    lines = [
        "# Regular vs ML Scheduler Comparison",
        "",
        f"- Comparison possible: Yes",
        f"- Evaluation type: **{evaluation_type}**",
        f"- Scenarios compared: {summary['scenario_count']}",
        f"- Mean regular actual time: {summary['mean_regular_actual_time']:.6f}",
        f"- Mean ML-selected actual time: {summary['mean_ml_selected_actual_time']:.6f}",
        f"- Mean percent improvement: {summary['mean_percent_improvement']:.6f}%",
        f"- Median percent improvement: "
        f"{summary['median_percent_improvement']:.6f}%",
        f"- ML wins: {summary['ml_win_count']}",
        f"- Regular scheduler wins: {summary['regular_scheduler_win_count']}",
        f"- Ties: {summary['tie_count']}",
        "",
        "The comparison is regular scheduler actual/simulated `workflow_time` versus "
        "ML-selected schedule actual/simulated `workflow_time`. Predicted time is "
        "used only to select a candidate, never as the final comparison outcome.",
        "",
        "## Main limitation",
        "",
        limitation,
        "",
        "## By workload type",
        "",
    ]
    lines.extend(markdown_table(summary["by_workload_type"], "workload_type"))
    lines.extend(["", "## By node count", ""])
    lines.extend(markdown_table(summary["by_node_count"], "node_count"))
    lines.extend(["", "## By environment condition", ""])
    lines.extend(
        markdown_table(
            summary["by_environment_condition"], "environment_condition"
        )
    )
    lines.append("")
    (output_dir / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# ML Scheduler Comparison",
                "",
                f"Comparison completed using **{evaluation_type}**.",
                "",
                limitation,
                "",
                "See `summary_report.md` and `comparison_results.csv`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def save_impossible(output_dir: Path, reasons: list[str]) -> None:
    message = (
        "Fair regular-vs-ML scheduling comparison is not possible with the current "
        "dataset because comparable regular and ML candidate schedules are not "
        "identifiable."
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=RESULT_COLUMNS).to_csv(
        output_dir / "comparison_results.csv", index=False
    )
    summary = {
        "comparison_possible": False,
        "evaluation_type": None,
        "scenario_count": 0,
        "reasons": reasons,
    }
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = ["# ML Scheduler Comparison", "", message, "", "## Reasons", ""]
    lines.extend(f"- {reason}" for reason in reasons)
    lines.append("")
    content = "\n".join(lines)
    (output_dir / "README.md").write_text(content, encoding="utf-8")
    (output_dir / "summary_report.md").write_text(content, encoding="utf-8")
    print(message)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline, _ = load_model(args.model_path)

    evaluation_type = None
    limitation = None
    results = None

    if args.mode in {"auto", "simulation"} and simulator_available(
        args.simulation_dir, args.workflows
    ):
        results = run_simulation_comparison(pipeline, args.simulation_dir, args)
        evaluation_type = "true simulation-based comparison"
        limitation = (
            "This is a deterministic simulator comparison, not a live Kubernetes "
            "deployment. The regular scheduler is the repository's existing default "
            "random placement generator, and the ML scheduler chooses from a finite "
            "random candidate pool under the same simulated node snapshot."
        )
    elif args.mode == "simulation":
        missing = [
            str(path)
            for path in required_simulation_files(args.simulation_dir, args.workflows)
            if not path.is_file()
        ]
        save_impossible(
            args.output_dir,
            ["Required simulator inputs are missing: " + ", ".join(missing)],
        )
        return

    if results is None and args.mode in {"auto", "offline"}:
        results = run_offline_comparison(pipeline, args.dataset_path)
        if results is not None:
            evaluation_type = "offline re-ranking evaluation"
            limitation = (
                "This evaluation re-ranks already observed candidates and does not "
                "run a scheduler or simulator. It is valid only for the identifiable "
                "candidate sets present in the source data."
            )

    if results is None or results.empty:
        save_impossible(
            args.output_dir,
            [
                "No usable live simulator path was selected or available.",
                "The extracted dataset does not contain identifiable regular and "
                "alternative candidates within the same source-file scenario.",
            ],
        )
        return

    results.to_csv(args.output_dir / "comparison_results.csv", index=False)
    summary = build_summary(results, evaluation_type)
    summary["random_state"] = args.random_state
    (args.output_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_plots(results, args.output_dir)
    save_success_report(args.output_dir, summary, evaluation_type, limitation)

    print(f"Comparison possible: yes ({evaluation_type})")
    print(f"Scenarios compared: {summary['scenario_count']}")
    print(f"Mean improvement: {summary['mean_percent_improvement']:.4f}%")
    print(f"ML win rate: {summary['ml_win_rate_percent']:.2f}%")
    print(f"Main limitation: {limitation}")


if __name__ == "__main__":
    main()
