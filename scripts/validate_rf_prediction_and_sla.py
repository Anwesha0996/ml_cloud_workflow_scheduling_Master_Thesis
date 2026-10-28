#!/usr/bin/env python3
"""Validate Random Forest workflow-time and SLA prediction robustness."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "outputs" / "feature_audit" / "extracted_ml_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ml_validation"

TARGET = "workflow_time"
SLA_PERCENTILE = 30
TEST_SIZE = 0.20
RANDOM_STATE = 42

SAFE_FEATURE_CANDIDATES = [
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

LEAKAGE_COLUMNS = [
    "workflow_time",
    "processing_time",
    "communication_delay",
    "memory_use",
    "communication_fraction",
    "sla_ok",
    "predicted_workflow_time",
    "predicted_sla_ok",
]

PROVENANCE_COLUMNS = [
    "source_file",
    "scenario_id",
    "scheduler_type",
    "candidate_id",
]


def make_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build the requested numeric and categorical preprocessing."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def make_regression_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    return Pipeline(
        [
            (
                "preprocessor",
                make_preprocessor(numeric_features, categorical_features),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_classification_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    return Pipeline(
        [
            (
                "preprocessor",
                make_preprocessor(numeric_features, categorical_features),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def compute_sla_thresholds(
    train_frame: pd.DataFrame,
) -> tuple[dict[str, float], float]:
    """Compute workflow-specific SLA thresholds from training targets only."""
    global_threshold = float(
        np.percentile(train_frame[TARGET], SLA_PERCENTILE)
    )
    thresholds: dict[str, float] = {}
    if "workflow_type" in train_frame.columns:
        known_workflows = train_frame.dropna(subset=["workflow_type"])
        for workflow, group in known_workflows.groupby("workflow_type"):
            thresholds[str(workflow)] = float(
                np.percentile(group[TARGET], SLA_PERCENTILE)
            )
    return thresholds, global_threshold


def create_sla_labels(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
    global_threshold: float,
) -> pd.Series:
    """Label rows using workflow thresholds, falling back to the global threshold."""
    if "workflow_type" not in frame.columns:
        row_thresholds = pd.Series(global_threshold, index=frame.index)
    else:
        row_thresholds = frame["workflow_type"].map(thresholds)
        row_thresholds = row_thresholds.fillna(global_threshold)
    return frame[TARGET].le(row_thresholds).astype(int)


def representative_examples(
    validation_frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Choose the validation row nearest median target for each workflow."""
    records = []
    for workflow in ("video", "iot", "cicd"):
        workflow_rows = validation_frame[
            validation_frame["workflow_type"].astype(str).str.lower().eq(workflow)
        ]
        if workflow_rows.empty:
            continue
        median_time = workflow_rows[TARGET].median()
        index = (workflow_rows[TARGET] - median_time).abs().idxmin()
        row = validation_frame.loc[index]
        input_values = {
            feature: (
                None
                if pd.isna(row[feature])
                else row[feature].item()
                if isinstance(row[feature], np.generic)
                else row[feature]
            )
            for feature in feature_columns
        }
        records.append(
            {
                "workflow_type": workflow,
                "actual_workflow_time": float(row[TARGET]),
                "predicted_workflow_time": float(
                    row["predicted_workflow_time"]
                ),
                "absolute_error": float(row["absolute_error"]),
                "actual_sla_ok": int(row["sla_ok"]),
                "predicted_sla_ok": int(row["predicted_sla_ok"]),
                "predicted_sla_probability": float(
                    row["predicted_sla_probability"]
                ),
                "input_feature_values": json.dumps(
                    input_values, sort_keys=True
                ),
            }
        )
    return pd.DataFrame(records)


def distribution_statistics(values: np.ndarray) -> dict[str, float]:
    """Calculate the requested workflow-time distribution statistics."""
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "percentile_25": float(np.percentile(values, 25)),
        "percentile_75": float(np.percentile(values, 75)),
    }


def prediction_distribution_metrics(
    actual_time: np.ndarray,
    predicted_time: np.ndarray,
) -> dict[str, Any]:
    """Compare actual and predicted validation distributions."""
    actual = distribution_statistics(actual_time)
    predicted = distribution_statistics(predicted_time)
    mean_error = predicted["mean"] - actual["mean"]
    median_error = predicted["median"] - actual["median"]
    return {
        "actual_workflow_time": actual,
        "predicted_workflow_time": predicted,
        "mean_error": mean_error,
        "median_error": median_error,
        "mean_percent_error": (
            mean_error / actual["mean"] * 100.0
            if not np.isclose(actual["mean"], 0.0)
            else None
        ),
        "median_percent_error": (
            median_error / actual["median"] * 100.0
            if not np.isclose(actual["median"], 0.0)
            else None
        ),
    }


def distributions_are_close(distribution: dict[str, Any]) -> bool:
    """Use a documented 5% relative-error rule for mean and median closeness."""
    mean_percent_error = distribution["mean_percent_error"]
    median_percent_error = distribution["median_percent_error"]
    return (
        mean_percent_error is not None
        and median_percent_error is not None
        and abs(mean_percent_error) <= 5.0
        and abs(median_percent_error) <= 5.0
    )


def save_distribution_plots(
    actual_time: np.ndarray,
    predicted_time: np.ndarray,
    distribution: dict[str, Any],
) -> None:
    """Save histogram and predicted-vs-actual scatter diagnostics."""
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.hist(
        actual_time,
        bins=50,
        alpha=0.55,
        label="Actual workflow_time",
        color="#777777",
    )
    axis.hist(
        predicted_time,
        bins=50,
        alpha=0.55,
        label="Predicted workflow_time",
        color="#4472C4",
    )
    axis.axvline(
        distribution["actual_workflow_time"]["median"],
        color="#333333",
        linestyle="--",
        linewidth=2,
        label="Actual median",
    )
    axis.axvline(
        distribution["predicted_workflow_time"]["median"],
        color="#1F4E99",
        linestyle=":",
        linewidth=2,
        label="Predicted median",
    )
    axis.set_xlabel("Workflow time")
    axis.set_ylabel("Validation sample count")
    axis.set_title("Actual vs Predicted Workflow-Time Distribution")
    axis.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "actual_vs_predicted_distribution.png",
        dpi=180,
    )
    plt.close(fig)

    plot_indices = np.arange(len(actual_time))
    if len(plot_indices) > 30_000:
        rng = np.random.default_rng(RANDOM_STATE)
        plot_indices = rng.choice(plot_indices, size=30_000, replace=False)
    plot_actual = actual_time[plot_indices]
    plot_predicted = predicted_time[plot_indices]
    lower = min(float(plot_actual.min()), float(plot_predicted.min()))
    upper = max(float(plot_actual.max()), float(plot_predicted.max()))

    fig, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(
        plot_actual,
        plot_predicted,
        alpha=0.18,
        s=10,
        edgecolors="none",
        color="#4472C4",
    )
    axis.plot([lower, upper], [lower, upper], color="black", linestyle="--")
    axis.set_xlabel("Actual workflow_time")
    axis.set_ylabel("Predicted workflow_time")
    axis.set_title("Actual vs Predicted Workflow Time")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "actual_vs_predicted_scatter.png",
        dpi=180,
    )
    plt.close(fig)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def interpretation_lines(
    regression_metrics: dict[str, float],
    classification_metrics: dict[str, float],
) -> list[str]:
    lines = []
    if regression_metrics["r2"] < 0.5:
        lines.append(
            "- R2 is low, so this Random Forest is not predicting "
            "`workflow_time` reliably."
        )
    elif regression_metrics["r2"] < 0.75:
        lines.append(
            "- R2 is moderate. The model captures useful signal, but "
            "`workflow_time` predictions are not yet strongly reliable."
        )
    else:
        lines.append(
            "- R2 is relatively strong on this validation split, though robustness "
            "still requires scenario- or source-grouped validation."
        )

    if classification_metrics["recall"] < 0.7:
        lines.append(
            "- SLA-positive recall is low, so the classifier misses many placements "
            "that actually satisfy the SLA."
        )
    if classification_metrics["sla_violation_recall"] < 0.7:
        lines.append(
            "- SLA-violation recall is low, so the classifier may miss bad "
            "placements that violate the SLA."
        )
    if (
        regression_metrics["r2"] < 0.75
        or classification_metrics["f1_score"] < 0.8
        or classification_metrics["sla_violation_recall"] < 0.8
    ):
        lines.append(
            "- Validation performance is not strong enough to claim that the ML "
            "scheduler is robust."
        )
    else:
        lines.append(
            "- Validation is encouraging, but this random row split alone is not "
            "enough to claim robust scheduler performance."
        )
    lines.append(
        "- This is model validation only. It does not train the final scheduler or "
        "compare ML scheduling against the regular scheduler."
    )
    return lines


def format_feature_values(raw_json: str) -> str:
    values = json.loads(raw_json)
    return ", ".join(f"`{key}={value}`" for key, value in values.items())


def write_report(
    feature_columns: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    thresholds: dict[str, float],
    global_threshold: float,
    regression_metrics: dict[str, float],
    classification_metrics: dict[str, Any],
    prediction_distribution: dict[str, Any],
    examples: pd.DataFrame,
) -> None:
    lines = [
        "# Random Forest Prediction and SLA Validation",
        "",
        "## Validation setup",
        "",
        f"- Rows: train/validation split with `test_size={TEST_SIZE}` and "
        f"`random_state={RANDOM_STATE}`",
        "- Split: ordinary random split, stratified by `workflow_type` when present",
        f"- SLA percentile: {SLA_PERCENTILE}",
        "- SLA thresholds: computed from training data only",
        "",
        "> **TODO:** Test a scenario-based or source-file-grouped split later. "
        "This random row split may place related observations in both train and "
        "validation sets and can therefore be optimistic.",
        "",
        "## Features",
        "",
        "**Features used:** "
        + ", ".join(f"`{feature}`" for feature in feature_columns),
        "",
        "**Numeric preprocessing:** median imputation for "
        + ", ".join(f"`{feature}`" for feature in numeric_features),
        "",
        "**Categorical preprocessing:** most-frequent imputation and one-hot "
        "encoding for "
        + ", ".join(f"`{feature}`" for feature in categorical_features),
        "",
        "**Excluded leakage columns:** "
        + ", ".join(f"`{column}`" for column in LEAKAGE_COLUMNS),
        "",
        "## SLA thresholds",
        "",
        f"- Global fallback: `{global_threshold:.6f}`",
    ]
    lines.extend(
        f"- `{workflow}`: `{threshold:.6f}`"
        for workflow, threshold in sorted(thresholds.items())
    )
    lines.extend(
        [
            "",
            "## Workflow-time regression",
            "",
            f"- MAE: `{regression_metrics['mae']:.6f}`",
            f"- RMSE: `{regression_metrics['rmse']:.6f}`",
            f"- R2: `{regression_metrics['r2']:.6f}`",
            f"- Mean actual workflow time: "
            f"`{regression_metrics['mean_actual_workflow_time']:.6f}`",
            f"- Mean predicted workflow time: "
            f"`{regression_metrics['mean_predicted_workflow_time']:.6f}`",
            "",
            "## Actual vs Predicted Workflow-Time Distribution",
            "",
            "| Statistic | Actual workflow_time | Predicted workflow_time | Difference |",
            "|---|---:|---:|---:|",
        ]
    )
    distribution_rows = [
        ("Mean", "mean"),
        ("Median", "median"),
        ("Standard deviation", "std"),
        ("Minimum", "min"),
        ("Maximum", "max"),
        ("25th percentile", "percentile_25"),
        ("75th percentile", "percentile_75"),
    ]
    actual_distribution = prediction_distribution["actual_workflow_time"]
    predicted_distribution = prediction_distribution["predicted_workflow_time"]
    for label, key in distribution_rows:
        actual_value = actual_distribution[key]
        predicted_value = predicted_distribution[key]
        lines.append(
            f"| {label} | {actual_value:.6f} | {predicted_value:.6f} | "
            f"{predicted_value - actual_value:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- Mean error (predicted - actual): "
            f"`{prediction_distribution['mean_error']:.6f}`",
            f"- Median error (predicted - actual): "
            f"`{prediction_distribution['median_error']:.6f}`",
            f"- Mean percent error: "
            f"`{prediction_distribution['mean_percent_error']:.6f}%`",
            f"- Median percent error: "
            f"`{prediction_distribution['median_percent_error']:.6f}%`",
            "",
        ]
    )
    if distributions_are_close(prediction_distribution):
        lines.append(
            "The RF model captures the overall workflow-time distribution "
            "reasonably well."
        )
    else:
        lines.append(
            "The RF model shows distribution mismatch, so individual predictions "
            "or scheduler selection may be unreliable."
        )
    lines.extend(
        [
            "",
            "> **Warning:** Matching the mean and median does not prove the "
            "scheduler selects the best placement. It only shows that the prediction "
            "distribution is close to the actual validation distribution.",
            "",
            "## SLA classification",
            "",
            f"- Accuracy: `{classification_metrics['accuracy']:.6f}`",
            f"- Precision: `{classification_metrics['precision']:.6f}`",
            f"- Recall (`sla_ok=1`): `{classification_metrics['recall']:.6f}`",
            f"- F1-score: `{classification_metrics['f1_score']:.6f}`",
            f"- SLA-violation recall (`sla_ok=0`): "
            f"`{classification_metrics['sla_violation_recall']:.6f}`",
            f"- Validation SLA-positive samples: "
            f"`{classification_metrics['positive_samples']}`",
            f"- Validation SLA-negative samples: "
            f"`{classification_metrics['negative_samples']}`",
            f"- Confusion matrix `[[TN, FP], [FN, TP]]`: "
            f"`{classification_metrics['confusion_matrix']}`",
            "",
            "## Representative validation cases",
            "",
        ]
    )
    if examples.empty:
        lines.append("No requested workflow examples were available.")
    else:
        for example in examples.itertuples(index=False):
            display_name = {
                "video": "Video",
                "iot": "IoT",
                "cicd": "CI/CD",
            }.get(example.workflow_type, example.workflow_type)
            lines.extend(
                [
                    f"### {display_name}",
                    "",
                    f"- Actual workflow time: "
                    f"`{example.actual_workflow_time:.6f}`",
                    f"- Predicted workflow time: "
                    f"`{example.predicted_workflow_time:.6f}`",
                    f"- Absolute error: `{example.absolute_error:.6f}`",
                    f"- Actual `sla_ok`: `{example.actual_sla_ok}`",
                    f"- Predicted `sla_ok`: `{example.predicted_sla_ok}`",
                    f"- Predicted SLA probability: "
                    f"`{example.predicted_sla_probability:.6f}`",
                    f"- Input features: "
                    f"{format_feature_values(example.input_feature_values)}",
                    "",
                ]
            )
    lines.extend(
        [
            "## Honest interpretation",
            "",
        ]
    )
    lines.extend(
        interpretation_lines(regression_metrics, classification_metrics)
    )
    lines.extend(
        [
            "",
            "## Run",
            "",
            "```bash",
            "python scripts/validate_rf_prediction_and_sla.py",
            "```",
            "",
        ]
    )
    (OUTPUT_DIR / "rf_validation_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {INPUT_PATH}.\n"
            "First run:\n"
            "python scripts/audit_and_extract_features.py"
        )

    frame = pd.read_csv(INPUT_PATH, low_memory=False)
    if TARGET not in frame.columns:
        raise ValueError(
            f"Required target column '{TARGET}' is missing from {INPUT_PATH}."
        )

    feature_columns = [
        column
        for column in SAFE_FEATURE_CANDIDATES
        if column in frame.columns and column not in LEAKAGE_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("No safe candidate feature columns are available.")

    model_frame = frame.dropna(subset=[TARGET]).copy()
    if model_frame.empty:
        raise ValueError("No rows have a non-missing workflow_time target.")

    categorical_features = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(model_frame[column])
    ]
    numeric_features = [
        column for column in feature_columns if column not in categorical_features
    ]

    stratify = None
    if "workflow_type" in model_frame.columns:
        workflow_strata = model_frame["workflow_type"].fillna("__missing__")
        if workflow_strata.value_counts().min() >= 2:
            stratify = workflow_strata

    # TODO: Compare this random split with scenario/source-file grouped validation.
    train_frame, validation_frame = train_test_split(
        model_frame,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    train_frame = train_frame.copy()
    validation_frame = validation_frame.copy()

    thresholds, global_threshold = compute_sla_thresholds(train_frame)
    train_frame["sla_ok"] = create_sla_labels(
        train_frame, thresholds, global_threshold
    )
    validation_frame["sla_ok"] = create_sla_labels(
        validation_frame, thresholds, global_threshold
    )

    regressor = make_regression_pipeline(
        numeric_features, categorical_features
    )
    classifier = make_classification_pipeline(
        numeric_features, categorical_features
    )
    regressor.fit(train_frame[feature_columns], train_frame[TARGET])
    classifier.fit(train_frame[feature_columns], train_frame["sla_ok"])

    predicted_time = regressor.predict(validation_frame[feature_columns])
    predicted_sla = classifier.predict(validation_frame[feature_columns])
    if hasattr(classifier, "predict_proba"):
        class_labels = classifier.named_steps["model"].classes_
        positive_index = int(np.where(class_labels == 1)[0][0])
        predicted_sla_probability = classifier.predict_proba(
            validation_frame[feature_columns]
        )[:, positive_index]
    else:
        predicted_sla_probability = np.full(len(validation_frame), np.nan)

    actual_time = validation_frame[TARGET].to_numpy()
    actual_sla = validation_frame["sla_ok"].to_numpy()
    matrix = confusion_matrix(actual_sla, predicted_sla, labels=[0, 1])

    regression_metrics = {
        "mae": float(mean_absolute_error(actual_time, predicted_time)),
        "rmse": float(np.sqrt(mean_squared_error(actual_time, predicted_time))),
        "r2": float(r2_score(actual_time, predicted_time)),
        "mean_actual_workflow_time": float(np.mean(actual_time)),
        "mean_predicted_workflow_time": float(np.mean(predicted_time)),
    }
    prediction_distribution = prediction_distribution_metrics(
        actual_time, predicted_time
    )
    classification_metrics = {
        "accuracy": float(accuracy_score(actual_sla, predicted_sla)),
        "precision": float(
            precision_score(actual_sla, predicted_sla, zero_division=0)
        ),
        "recall": float(
            recall_score(actual_sla, predicted_sla, zero_division=0)
        ),
        "f1_score": float(
            f1_score(actual_sla, predicted_sla, zero_division=0)
        ),
        "sla_violation_recall": float(
            recall_score(
                actual_sla,
                predicted_sla,
                pos_label=0,
                zero_division=0,
            )
        ),
        "confusion_matrix": matrix.tolist(),
        "positive_samples": int(np.sum(actual_sla == 1)),
        "negative_samples": int(np.sum(actual_sla == 0)),
    }

    validation_frame["predicted_workflow_time"] = predicted_time
    validation_frame["actual_workflow_time"] = actual_time
    validation_frame["prediction_error"] = predicted_time - actual_time
    validation_frame["absolute_error"] = np.abs(
        validation_frame["prediction_error"]
    )
    validation_frame["percent_error"] = np.where(
        np.isclose(actual_time, 0.0),
        np.nan,
        validation_frame["prediction_error"] / actual_time * 100.0,
    )
    validation_frame["predicted_sla_ok"] = predicted_sla
    validation_frame["predicted_sla_probability"] = predicted_sla_probability

    output_columns = [
        column for column in PROVENANCE_COLUMNS if column in validation_frame.columns
    ]
    output_columns.extend(feature_columns)
    output_columns.extend(
        [
            TARGET,
            "actual_workflow_time",
            "predicted_workflow_time",
            "prediction_error",
            "absolute_error",
            "percent_error",
            "sla_ok",
            "predicted_sla_ok",
            "predicted_sla_probability",
        ]
    )
    validation_predictions = validation_frame[output_columns].copy()
    examples = representative_examples(validation_frame, feature_columns)

    confusion_frame = pd.DataFrame(
        matrix,
        index=["actual_sla_violation_0", "actual_sla_ok_1"],
        columns=["predicted_sla_violation_0", "predicted_sla_ok_1"],
    )
    confusion_frame.index.name = "actual_label"

    metrics = {
        "validation_type": (
            "random row split; scenario/source-file grouping not applied"
        ),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "sla_percentile": SLA_PERCENTILE,
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "features_used": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "excluded_leakage_columns": LEAKAGE_COLUMNS,
        "sla_thresholds_by_workflow": thresholds,
        "global_sla_threshold": global_threshold,
        "regression": regression_metrics,
        "classification": classification_metrics,
        "prediction_distribution": prediction_distribution,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_distribution_plots(
        actual_time,
        predicted_time,
        prediction_distribution,
    )
    (OUTPUT_DIR / "rf_validation_metrics.json").write_text(
        json.dumps(json_safe(metrics), indent=2), encoding="utf-8"
    )
    validation_predictions.to_csv(
        OUTPUT_DIR / "validation_predictions.csv", index=False
    )
    examples.to_csv(
        OUTPUT_DIR / "example_cases_by_workflow.csv", index=False
    )
    confusion_frame.to_csv(OUTPUT_DIR / "sla_confusion_matrix.csv")
    write_report(
        feature_columns,
        numeric_features,
        categorical_features,
        thresholds,
        global_threshold,
        regression_metrics,
        classification_metrics,
        prediction_distribution,
        examples,
    )

    print("Random Forest validation completed.")
    print(f"Validation rows: {len(validation_frame):,}")
    print(f"Regression R2: {regression_metrics['r2']:.4f}")
    print(f"Regression MAE: {regression_metrics['mae']:.4f}")
    print(f"SLA F1-score: {classification_metrics['f1_score']:.4f}")
    print(f"SLA violation recall: {classification_metrics['sla_violation_recall']:.4f}")
    print(f"Outputs: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
