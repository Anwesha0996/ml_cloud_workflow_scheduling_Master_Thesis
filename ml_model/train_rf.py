#!/usr/bin/env python3
"""Train a leakage-safe Random Forest workflow-time regression baseline."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "outputs" / "feature_audit" / "extracted_ml_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rf_regression"

TARGET = "workflow_time"
CATEGORICAL_FEATURES = [
    "workflow_type",
    "workload_type",
    "environment_condition",
]
NUMERIC_FEATURES = [
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
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

TEST_SIZE = 0.20
RANDOM_STATE = 42


def make_pipeline() -> Pipeline:
    """Create preprocessing and Random Forest regression steps."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    regressor = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features=1.0,
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def save_feature_importance(model: Pipeline) -> pd.DataFrame:
    """Save encoded and source-feature Random Forest importances."""
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]
    encoded_names = preprocessor.get_feature_names_out()

    encoded = pd.DataFrame(
        {
            "encoded_feature": encoded_names,
            "importance": regressor.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    encoded.to_csv(OUTPUT_DIR / "encoded_feature_importance.csv", index=False)

    source_feature = {feature: feature for feature in NUMERIC_FEATURES}
    encoder = preprocessor.named_transformers_["categorical"]
    start = 0
    for feature, categories in zip(CATEGORICAL_FEATURES, encoder.categories_):
        for encoded_name in encoded_names[start : start + len(categories)]:
            source_feature[encoded_name] = feature
        start += len(categories)

    encoded["feature"] = encoded["encoded_feature"].map(source_feature)
    aggregate = (
        encoded.groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )
    aggregate.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    return aggregate


def save_importance_plot(importance: pd.DataFrame) -> None:
    """Plot aggregate importance for the 12 source features."""
    ordered = importance.sort_values("importance")
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.barh(ordered["feature"], ordered["importance"], color="#4472C4")
    axis.set_xlabel("Random Forest importance")
    axis.set_title("Workflow-Time Feature Importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=180)
    plt.close(fig)


def save_prediction_plots(predictions: pd.DataFrame) -> None:
    """Save predicted-vs-actual and residual diagnostic plots."""
    plot_data = predictions
    if len(plot_data) > 30_000:
        plot_data = plot_data.sample(30_000, random_state=RANDOM_STATE)

    actual = plot_data["actual_workflow_time"]
    predicted = plot_data["predicted_workflow_time"]
    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())

    fig, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(actual, predicted, alpha=0.18, s=10, edgecolors="none")
    axis.plot([lower, upper], [lower, upper], color="black", linestyle="--")
    axis.set_xlabel("Actual workflow_time")
    axis.set_ylabel("Predicted workflow_time")
    axis.set_title("Predicted vs Actual Workflow Time")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "predicted_vs_actual.png", dpi=180)
    plt.close(fig)

    residual = actual - predicted
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(predicted, residual, alpha=0.18, s=10, edgecolors="none")
    axis.axhline(0, color="black", linestyle="--")
    axis.set_xlabel("Predicted workflow_time")
    axis.set_ylabel("Residual (actual - predicted)")
    axis.set_title("Random Forest Residuals")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "residuals_vs_predicted.png", dpi=180)
    plt.close(fig)


def save_report(metrics: dict[str, object], importance: pd.DataFrame) -> None:
    """Write a concise Markdown training report."""
    lines = [
        "# Random Forest Regression Baseline",
        "",
        "## Split strategy",
        "",
        f"- Strategy: {metrics['split_strategy']}",
        f"- Training rows: {metrics['train_rows']}",
        f"- Test rows: {metrics['test_rows']}",
        f"- Complete rows used: {metrics['complete_rows']}",
        f"- Rows dropped for missing values: {metrics['rows_dropped']}",
        f"- Exact duplicate feature/target rows: {metrics['duplicate_rows']}",
        "",
        f"- Training source-file groups: {metrics['train_groups']}",
        f"- Test source-file groups: {metrics['test_groups']}",
        f"- Source-file group overlap: {metrics['group_overlap']}",
        "",
        "Source-file identifiers are used only for splitting and traceability, "
        "never as model inputs.",
        "",
        "## Test metrics",
        "",
        f"- MAE: {metrics['mae']:.6f}",
        f"- RMSE: {metrics['rmse']:.6f}",
        f"- R2: {metrics['r2']:.6f}",
        "",
        "## Aggregate feature importance",
        "",
        "| Feature | Importance |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{row.feature}` | {row.importance:.6f} |"
        for row in importance.itertuples(index=False)
    )
    lines.extend(
        [
            "",
            "The model uses only the 12 requested leakage-safe features. "
            "`avg_node_cost` and `max_node_cost` are excluded.",
            "",
            "## Artifacts",
            "",
            "- `rf_regression_pipeline.joblib`",
            "- `metrics.json`",
            "- `test_predictions.csv`",
            "- `feature_importance.csv`",
            "- `encoded_feature_importance.csv`",
            "- `feature_importance.png`",
            "- `predicted_vs_actual.png`",
            "- `residuals_vs_predicted.png`",
            "",
        ]
    )
    (OUTPUT_DIR / "training_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Load clean rows, train the baseline, and save evaluation artifacts."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run scripts/audit_and_extract_features.py first."
        )

    frame = pd.read_csv(DATA_PATH, low_memory=False)
    missing_columns = [
        column for column in FEATURES + [TARGET] if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"Extracted dataset is missing columns: {missing_columns}")

    provenance_columns = [
        column
        for column in ["source_file", "scenario_id", "scheduler_type", "candidate_id"]
        if column in frame.columns
    ]
    model_frame = frame[provenance_columns + FEATURES + [TARGET]].dropna(
        subset=FEATURES + [TARGET]
    ).copy()
    if model_frame.empty:
        raise ValueError("No complete rows are available for the requested features.")

    duplicate_rows = int(
        model_frame[FEATURES + [TARGET]].duplicated(keep=False).sum()
    )
    X = model_frame[FEATURES]
    y = model_frame[TARGET]

    if "source_file" in model_frame.columns and model_frame["source_file"].nunique() > 1:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
        train_indices, test_indices = next(
            splitter.split(X, y, groups=model_frame["source_file"])
        )
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]
        train_groups = set(model_frame.iloc[train_indices]["source_file"])
        test_groups = set(model_frame.iloc[test_indices]["source_file"])
        split_strategy = "grouped 80/20 split by source_file"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
        train_groups = set()
        test_groups = set()
        split_strategy = "random 80/20 split (baseline only; no group identifier)"

    model = make_pipeline()
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    mae = mean_absolute_error(y_test, predicted)
    rmse = np.sqrt(mean_squared_error(y_test, predicted))
    r2 = r2_score(y_test, predicted)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        {
            "actual_workflow_time": y_test.to_numpy(),
            "predicted_workflow_time": predicted,
            "residual": y_test.to_numpy() - predicted,
        },
        index=y_test.index,
    )
    for column in provenance_columns:
        predictions.insert(
            len(predictions.columns) - 3,
            column,
            model_frame.loc[y_test.index, column].to_numpy(),
        )
    predictions.index.name = "source_row_index"
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv")

    importance = save_feature_importance(model)
    save_importance_plot(importance)
    save_prediction_plots(predictions)

    metrics: dict[str, object] = {
        "split_strategy": split_strategy,
        "random_state": RANDOM_STATE,
        "total_input_rows": int(len(frame)),
        "complete_rows": int(len(model_frame)),
        "rows_dropped": int(len(frame) - len(model_frame)),
        "duplicate_rows": duplicate_rows,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_groups": len(train_groups),
        "test_groups": len(test_groups),
        "group_overlap": len(train_groups & test_groups),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "excluded_features": ["avg_node_cost", "max_node_cost"],
        "target": TARGET,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    save_report(metrics, importance)

    joblib.dump(
        {
            "pipeline": model,
            "features": FEATURES,
            "target": TARGET,
            "metrics": metrics,
        },
        OUTPUT_DIR / "rf_regression_pipeline.joblib",
    )

    print("Random Forest regression baseline trained.")
    print(f"Complete rows: {len(model_frame):,}")
    print(f"Split: {split_strategy}")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R2:   {r2:.6f}")
    print(f"Artifacts: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
