#!/usr/bin/env python3
"""Audit training CSVs and build a leakage-aware ML feature dataset."""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "feature_audit"

LEAKAGE_COLUMNS = {
    "workflow_time",
    "processing_time",
    "communication_delay",
    "memory_use",
    "communication_fraction",
}

CANDIDATE_FEATURES = [
    "workflow_type",
    "node_count",
    "workload_type",
    "environment_condition",
    "communication_scale",
    "memory_scale",
    "num_tasks",
    "num_edges",
    "dag_depth",
    "distinct_nodes_used",
    "dag_crossings",
    "colocated_edges",
    "avg_node_cost",
    "max_node_cost",
]

IDENTIFIER_COLUMNS = [
    "source_file",
    "scenario_id",
    "scheduler_type",
    "candidate_id",
]
EXTRACTED_COLUMNS = IDENTIFIER_COLUMNS + CANDIDATE_FEATURES + ["workflow_time"]
PLACEMENT_FEATURES = ["distinct_nodes_used", "dag_crossings", "colocated_edges"]

WORKFLOW_TYPES = ("video", "iot", "cicd")
WORKLOAD_PATTERNS = (
    ("data_intensive", "data_intensive_workload"),
    ("light", "light_workload"),
    ("heavy", "heavy_workload"),
    ("mixed", "mixed_workload"),
)
ENVIRONMENT_CONDITIONS = (
    "heavy_memory_use",
    "light_memory_use",
    "fast_network",
    "slow_network",
    "normal",
)


def infer_assignment_workflow(columns: list[str]) -> str | None:
    """Infer workflow type from distinctive assigned task columns."""
    column_set = set(columns)
    if "assigned_node_take_video" in column_set:
        return "video"
    if "assigned_node_sensor_collector" in column_set:
        return "iot"
    if "assigned_node_source_fetch" in column_set:
        return "cicd"
    return None


def relative_path(path: Path) -> str:
    """Return a stable project-relative path for output files."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def parse_metadata(path: Path, columns: list[str]) -> dict[str, Any]:
    """Parse scenario metadata from the filename/path and task signatures."""
    searchable = "_".join(part.lower() for part in path.parts)
    stem = path.stem.lower()

    workflow_type = next(
        (
            workflow
            for workflow in WORKFLOW_TYPES
            if re.search(rf"(?:^|_){workflow}(?:_|$)", stem)
        ),
        None,
    )
    if workflow_type is None:
        workflow_type = next(
            (
                workflow
                for workflow in WORKFLOW_TYPES
                if any(
                    part.lower() in {workflow, f"results_{workflow}"}
                    for part in path.parts
                )
            ),
            None,
        )

    assignment_workflow_type = infer_assignment_workflow(columns)

    # Task signatures recover workflow metadata for generic simulation filenames.
    if workflow_type is None:
        workflow_type = assignment_workflow_type

    node_match = re.search(r"train_nodes_(\d+)", stem)
    seed_match = re.search(r"(?:^|_)seed(\d+)(?:_|$)", stem)

    workload_type = next(
        (
            label
            for label, filename_token in WORKLOAD_PATTERNS
            if filename_token in stem
        ),
        None,
    )
    environment_condition = next(
        (
            condition
            for condition in ENVIRONMENT_CONDITIONS
            if re.search(rf"(?:^|_){condition}(?:_|$)", searchable)
        ),
        None,
    )

    return {
        "workflow_type": workflow_type,
        "node_count": int(node_match.group(1)) if node_match else None,
        "workload_type": workload_type,
        "environment_condition": environment_condition,
        "seed": int(seed_match.group(1)) if seed_match else None,
        "assignment_workflow_type": assignment_workflow_type,
        "workflow_metadata_conflict": bool(
            workflow_type
            and assignment_workflow_type
            and workflow_type != assignment_workflow_type
        ),
    }


def longest_path_depth(edges: list[tuple[str, str]]) -> int | None:
    """Return DAG depth in edges using Kahn's algorithm, or None for a cycle."""
    nodes = {node for edge in edges for node in edge}
    if not nodes:
        return 0

    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree = {node: 0 for node in nodes}
    for source, target in edges:
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
        return None
    return max(distance.values(), default=0)


def read_dag(path: Path) -> dict[str, Any] | None:
    """Read one application DAG with flexible source/target column names."""
    try:
        dag_df = pd.read_csv(path)
    except Exception:
        return None

    lowered = {str(column).lower(): column for column in dag_df.columns}
    source_column = next(
        (lowered[name] for name in ("src", "source", "from") if name in lowered),
        None,
    )
    target_column = next(
        (lowered[name] for name in ("dst", "target", "to") if name in lowered),
        None,
    )
    if source_column is None or target_column is None:
        return None

    edge_frame = dag_df[[source_column, target_column]].dropna()
    edges = [
        (str(source).strip(), str(target).strip())
        for source, target in edge_frame.itertuples(index=False, name=None)
    ]
    tasks = {task for edge in edges for task in edge}
    return {
        "path": path,
        "edges": edges,
        "num_edges": len(edges),
        "num_tasks": len(tasks),
        "dag_depth": longest_path_depth(edges),
    }


def discover_dags() -> dict[str, list[dict[str, Any]]]:
    """Index all readable application DAG files by workflow type."""
    dags: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(PROJECT_ROOT.rglob("application_dag_*.csv")):
        match = re.fullmatch(r"application_dag_(video|iot|cicd)", path.stem.lower())
        if not match:
            continue
        dag = read_dag(path)
        if dag is not None:
            dags[match.group(1)].append(dag)
    return dags


def path_distance(left: Path, right: Path) -> int:
    """Count directory steps between two resolved paths."""
    left_parts = left.resolve().parts
    right_parts = right.resolve().parts
    common = 0
    for left_part, right_part in zip(left_parts, right_parts):
        if left_part != right_part:
            break
        common += 1
    return len(left_parts) + len(right_parts) - (2 * common)


def choose_dag(
    csv_path: Path,
    workflow_type: str | None,
    dag_index: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Choose the nearest DAG definition matching the parsed workflow."""
    if workflow_type is None or not dag_index.get(workflow_type):
        return None
    return min(
        dag_index[workflow_type],
        key=lambda dag: path_distance(csv_path.parent, dag["path"].parent),
    )


def numeric_summary(series: pd.Series) -> dict[str, Any]:
    """Calculate numeric statistics without coercing non-numeric columns."""
    if not pd.api.types.is_numeric_dtype(series):
        return {"std": None, "min": None, "max": None, "mean": None}

    numeric = pd.to_numeric(series, errors="coerce")
    return {
        "std": numeric.std(),
        "min": numeric.min(),
        "max": numeric.max(),
        "mean": numeric.mean(),
    }


def build_column_audit(path: Path, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Build one audit record per source column."""
    records = []
    row_count = len(frame)
    for column in frame.columns:
        series = frame[column]
        summary = numeric_summary(series)
        records.append(
            {
                "file_path": relative_path(path),
                "column_name": column,
                "dtype": str(series.dtype),
                "row_count": row_count,
                "missing_percentage": (
                    float(series.isna().mean() * 100) if row_count else 0.0
                ),
                "unique_values": int(series.nunique(dropna=True)),
                "is_numeric": pd.api.types.is_numeric_dtype(series),
                "std": summary["std"],
                "min": summary["min"],
                "max": summary["max"],
                "mean": summary["mean"],
                "is_target_leakage": column in LEAKAGE_COLUMNS,
                "safe_as_workflow_time_input": column not in LEAKAGE_COLUMNS,
            }
        )
    return records


def derive_placement_features(
    frame: pd.DataFrame,
    dag: dict[str, Any] | None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Derive placement counts, retaining existing metrics when provided."""
    index = frame.index
    assignment_columns = [
        column for column in frame.columns if column.startswith("assigned_node_")
    ]

    if "distinct_nodes_used" in frame.columns:
        distinct_nodes = pd.to_numeric(
            frame["distinct_nodes_used"], errors="coerce"
        )
    elif assignment_columns:
        distinct_nodes = frame[assignment_columns].nunique(axis=1, dropna=True)
        distinct_nodes = distinct_nodes.where(
            frame[assignment_columns].notna().any(axis=1), np.nan
        )
    else:
        distinct_nodes = pd.Series(np.nan, index=index, dtype=float)

    if "dag_crossings" in frame.columns:
        dag_crossings = pd.to_numeric(frame["dag_crossings"], errors="coerce")
    else:
        dag_crossings = pd.Series(np.nan, index=index, dtype=float)

    if "colocated_edges" in frame.columns:
        colocated_edges = pd.to_numeric(frame["colocated_edges"], errors="coerce")
    else:
        colocated_edges = pd.Series(np.nan, index=index, dtype=float)

    if dag is None or not assignment_columns:
        return distinct_nodes, dag_crossings, colocated_edges

    derived_crossings = pd.Series(0.0, index=index)
    derived_colocated = pd.Series(0.0, index=index)
    valid_edge_count = pd.Series(0, index=index)

    for source, target in dag["edges"]:
        source_column = f"assigned_node_{source}"
        target_column = f"assigned_node_{target}"
        if source_column not in frame.columns or target_column not in frame.columns:
            continue

        valid = frame[source_column].notna() & frame[target_column].notna()
        same_node = frame[source_column].eq(frame[target_column])
        derived_crossings += (valid & ~same_node).astype(float)
        derived_colocated += (valid & same_node).astype(float)
        valid_edge_count += valid.astype(int)

    complete_placement = valid_edge_count.eq(dag["num_edges"])
    if "dag_crossings" not in frame.columns:
        dag_crossings = derived_crossings.where(complete_placement, np.nan)
    if "colocated_edges" not in frame.columns:
        colocated_edges = derived_colocated.where(complete_placement, np.nan)

    return distinct_nodes, dag_crossings, colocated_edges


def series_or_missing(
    frame: pd.DataFrame,
    column: str,
    fallback: Any = None,
) -> pd.Series:
    """Return a source column or a same-length fallback series."""
    if column in frame.columns:
        return frame[column]
    return pd.Series(fallback, index=frame.index)


def extract_rows(
    path: Path,
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    dag: dict[str, Any] | None,
) -> pd.DataFrame:
    """Build the requested safe candidate feature rows plus the target."""
    distinct_nodes, dag_crossings, colocated_edges = derive_placement_features(
        frame, dag
    )

    file_environment = metadata["environment_condition"]
    environment = series_or_missing(frame, "environment_condition", file_environment)
    if file_environment is not None:
        environment = environment.fillna(file_environment)

    extracted = pd.DataFrame(index=frame.index)
    extracted["source_file"] = relative_path(path)
    extracted["scenario_id"] = series_or_missing(frame, "scenario_id")
    extracted["scheduler_type"] = series_or_missing(frame, "scheduler_type")
    extracted["candidate_id"] = series_or_missing(frame, "candidate_id")
    extracted["workflow_type"] = metadata["workflow_type"]
    extracted["node_count"] = metadata["node_count"]
    extracted["workload_type"] = metadata["workload_type"]
    extracted["environment_condition"] = environment
    extracted["communication_scale"] = pd.to_numeric(
        series_or_missing(frame, "communication_scale"), errors="coerce"
    )
    extracted["memory_scale"] = pd.to_numeric(
        series_or_missing(frame, "memory_scale"), errors="coerce"
    )
    extracted["num_tasks"] = dag["num_tasks"] if dag else np.nan
    extracted["num_edges"] = dag["num_edges"] if dag else np.nan
    extracted["dag_depth"] = dag["dag_depth"] if dag else np.nan
    extracted["distinct_nodes_used"] = distinct_nodes
    extracted["dag_crossings"] = dag_crossings
    extracted["colocated_edges"] = colocated_edges
    extracted["avg_node_cost"] = pd.to_numeric(
        series_or_missing(frame, "avg_node_cost"), errors="coerce"
    )
    extracted["max_node_cost"] = pd.to_numeric(
        series_or_missing(frame, "max_node_cost"), errors="coerce"
    )
    extracted["workflow_time"] = pd.to_numeric(
        series_or_missing(frame, "workflow_time"), errors="coerce"
    )
    return extracted[EXTRACTED_COLUMNS]


def build_report(
    train_files: list[Path],
    total_rows: int,
    extracted: pd.DataFrame,
    file_audit: pd.DataFrame,
    read_errors: list[tuple[str, str]],
) -> str:
    """Create the human-readable feature availability and safety report."""
    lines = [
        "# Feature Audit and Extraction Report",
        "",
        "## Run the pipeline",
        "",
        "From the project root:",
        "",
        "```bash",
        "python scripts/audit_and_extract_features.py",
        "```",
        "",
        "(`python3 scripts/audit_and_extract_features.py` is equivalent on systems "
        "where `python` is not configured.)",
        "",
        "## Scan summary",
        "",
        f"- Total files discovered: {len(train_files)}",
        f"- Files read successfully: {len(train_files) - len(read_errors)}",
        f"- Total rows scanned: {total_rows}",
        f"- Extracted dataset rows: {len(extracted)}",
        f"- Read errors: {len(read_errors)}",
        "",
        "The extracted dataset contains one row per successfully read source row. "
        "DAG depth is measured as the longest source-to-sink path in edges.",
        "",
        "`source_file`, `scenario_id`, `scheduler_type`, and `candidate_id` are "
        "provenance identifiers only. They are not recommended as model inputs.",
        "",
        "## Candidate feature availability and variation",
        "",
        "| Feature | Non-missing rows | Available % | Unique values | Assessment |",
        "|---|---:|---:|---:|---|",
    ]

    feature_stats: dict[str, dict[str, Any]] = {}
    for feature in CANDIDATE_FEATURES:
        series = extracted[feature]
        non_missing = int(series.notna().sum())
        available_percentage = (
            (non_missing / len(extracted) * 100) if len(extracted) else 0.0
        )
        unique_values = int(series.nunique(dropna=True))
        if non_missing == 0:
            assessment = "Unavailable"
        elif unique_values <= 1:
            assessment = "Low variation"
        elif available_percentage < 50:
            assessment = "Missing often"
        else:
            assessment = "Usable variation"
        feature_stats[feature] = {
            "non_missing": non_missing,
            "available_percentage": available_percentage,
            "unique_values": unique_values,
        }
        lines.append(
            f"| `{feature}` | {non_missing} | {available_percentage:.2f}% | "
            f"{unique_values} | {assessment} |"
        )

    available = [
        feature
        for feature, stats in feature_stats.items()
        if stats["non_missing"] > 0
    ]
    missing_often = [
        feature
        for feature, stats in feature_stats.items()
        if stats["available_percentage"] < 50
    ]
    low_variation = [
        feature
        for feature, stats in feature_stats.items()
        if stats["non_missing"] > 0 and stats["unique_values"] <= 1
    ]
    recommended = [
        feature
        for feature, stats in feature_stats.items()
        if stats["available_percentage"] >= 50 and stats["unique_values"] > 1
    ]

    lines.extend(
        [
            "",
            "## Availability findings",
            "",
            "**Available features:** "
            + (", ".join(f"`{feature}`" for feature in available) or "None"),
            "",
            "**Missing often (more than 50% missing):** "
            + (", ".join(f"`{feature}`" for feature in missing_often) or "None"),
            "",
            "**Low variation (zero or one observed value):** "
            + (", ".join(f"`{feature}`" for feature in low_variation) or "None"),
            "",
            "## Target leakage",
            "",
            "The following columns are unsafe as model inputs when predicting "
            "`workflow_time`:",
            "",
        ]
    )
    lines.extend(f"- `{column}`" for column in sorted(LEAKAGE_COLUMNS))
    lines.extend(
        [
            "",
            "`workflow_time` is retained only as the prediction target. The other "
            "leakage columns are excluded from `extracted_ml_dataset.csv`.",
            "",
            "## Recommended safe Random Forest features",
            "",
        ]
    )
    lines.extend(
        [f"- `{feature}`" for feature in recommended]
        or ["No feature currently satisfies the availability and variation rules."]
    )
    lines.extend(
        [
            "",
            "These recommendations exclude target leakage and require at least "
            "50% availability plus more than one observed value. Categorical "
            "features must be encoded before fitting a scikit-learn Random Forest.",
            "",
            "Placement features are leakage-safe with respect to measured runtime, "
            "but they are only valid predictors when a candidate placement is known "
            "at prediction time.",
            "",
            "## Placement feature coverage",
            "",
        ]
    )

    successful_files = file_audit[file_audit["read_error"].isna()]
    for feature in PLACEMENT_FEATURES:
        files_with_feature = int(successful_files[f"{feature}_available"].sum())
        lines.append(
            f"- `{feature}` is available in {files_with_feature} of "
            f"{len(successful_files)} successfully read files."
        )

    placement_partial = any(
        0 < int(successful_files[f"{feature}_available"].sum()) < len(successful_files)
        for feature in PLACEMENT_FEATURES
    )
    if placement_partial:
        lines.extend(
            [
                "",
                "> **Warning:** Placement features exist only in a subset of files. "
                "Training directly on all rows will require filtering, imputation, "
                "or separate model variants.",
            ]
        )

    metadata_conflicts = int(successful_files["workflow_metadata_conflict"].sum())
    lines.extend(
        [
            "",
            "## Metadata consistency",
            "",
            f"- Files whose filename/path workflow conflicts with assignment task "
            f"columns: {metadata_conflicts}",
        ]
    )
    if metadata_conflicts:
        lines.extend(
            [
                "",
                "> **Warning:** Conflicting files keep the filename/path "
                "`workflow_type`, but DAG placement metrics are left missing rather "
                "than being calculated against the wrong workflow. See "
                "`workflow_metadata_conflict` in `file_level_audit.csv`.",
            ]
        )

    if read_errors:
        lines.extend(["", "## Read errors", ""])
        lines.extend(
            f"- `{path}`: {message}" for path, message in read_errors
        )

    lines.extend(
        [
            "",
            "## Output files",
            "",
            "- `outputs/feature_audit/feature_column_audit.csv`",
            "- `outputs/feature_audit/file_level_audit.csv`",
            "- `outputs/feature_audit/extracted_ml_dataset.csv`",
            "- `outputs/feature_audit/usable_features_report.md`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Run the complete audit and extraction pipeline."""
    train_files = sorted(
        path
        for path in PROJECT_ROOT.rglob("train*.csv")
        if OUTPUT_DIR not in path.parents
    )
    dag_index = discover_dags()

    column_records: list[dict[str, Any]] = []
    file_records: list[dict[str, Any]] = []
    extracted_frames: list[pd.DataFrame] = []
    read_errors: list[tuple[str, str]] = []
    total_rows = 0

    for path in train_files:
        base_record: dict[str, Any] = {
            "file_path": relative_path(path),
            "row_count": None,
            "column_count": None,
            "column_names": None,
            "workflow_type": None,
            "node_count": None,
            "workload_type": None,
            "environment_condition": None,
            "seed": None,
            "assignment_workflow_type": None,
            "workflow_metadata_conflict": False,
            "dag_file": None,
            "num_tasks": None,
            "num_edges": None,
            "dag_depth": None,
            "assigned_node_column_count": None,
            "distinct_nodes_used_available": False,
            "dag_crossings_available": False,
            "colocated_edges_available": False,
            "read_error": None,
        }

        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            base_record["read_error"] = message
            file_records.append(base_record)
            read_errors.append((relative_path(path), message))
            continue

        metadata = parse_metadata(path, list(frame.columns))
        dag = choose_dag(path, metadata["workflow_type"], dag_index)
        extracted = extract_rows(path, frame, metadata, dag)

        total_rows += len(frame)
        column_records.extend(build_column_audit(path, frame))
        extracted_frames.append(extracted)

        base_record.update(metadata)
        base_record.update(
            {
                "row_count": len(frame),
                "column_count": len(frame.columns),
                "column_names": json.dumps(list(frame.columns)),
                "dag_file": relative_path(dag["path"]) if dag else None,
                "num_tasks": dag["num_tasks"] if dag else None,
                "num_edges": dag["num_edges"] if dag else None,
                "dag_depth": dag["dag_depth"] if dag else None,
                "assigned_node_column_count": sum(
                    column.startswith("assigned_node_")
                    for column in frame.columns
                ),
                "distinct_nodes_used_available": bool(
                    extracted["distinct_nodes_used"].notna().any()
                ),
                "dag_crossings_available": bool(
                    extracted["dag_crossings"].notna().any()
                ),
                "colocated_edges_available": bool(
                    extracted["colocated_edges"].notna().any()
                ),
            }
        )
        file_records.append(base_record)

    column_audit = pd.DataFrame(column_records)
    file_audit = pd.DataFrame(file_records)
    extracted_dataset = (
        pd.concat(extracted_frames, ignore_index=True)
        if extracted_frames
        else pd.DataFrame(columns=EXTRACTED_COLUMNS)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    column_audit.to_csv(OUTPUT_DIR / "feature_column_audit.csv", index=False)
    file_audit.to_csv(OUTPUT_DIR / "file_level_audit.csv", index=False)
    extracted_dataset.to_csv(OUTPUT_DIR / "extracted_ml_dataset.csv", index=False)

    report = build_report(
        train_files=train_files,
        total_rows=total_rows,
        extracted=extracted_dataset,
        file_audit=file_audit,
        read_errors=read_errors,
    )
    (OUTPUT_DIR / "usable_features_report.md").write_text(
        report, encoding="utf-8"
    )

    print(f"Scanned {len(train_files)} files and {total_rows} rows.")
    print(f"Wrote feature audit outputs to {relative_path(OUTPUT_DIR)}/")


if __name__ == "__main__":
    main()
