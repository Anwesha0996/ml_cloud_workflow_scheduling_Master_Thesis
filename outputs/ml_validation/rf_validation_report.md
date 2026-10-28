# Random Forest Prediction and SLA Validation

## Validation setup

- Rows: train/validation split with `test_size=0.2` and `random_state=42`
- Split: ordinary random split, stratified by `workflow_type` when present
- SLA percentile: 30
- SLA thresholds: computed from training data only

> **TODO:** Test a scenario-based or source-file-grouped split later. This random row split may place related observations in both train and validation sets and can therefore be optimistic.

## Features

**Features used:** `workflow_type`, `node_count`, `workload_type`, `environment_condition`, `communication_scale`, `memory_scale`, `num_tasks`, `num_edges`, `dag_depth`, `distinct_nodes_used`, `dag_crossings`, `colocated_edges`, `avg_node_cost`, `max_node_cost`

**Numeric preprocessing:** median imputation for `node_count`, `communication_scale`, `memory_scale`, `num_tasks`, `num_edges`, `dag_depth`, `distinct_nodes_used`, `dag_crossings`, `colocated_edges`, `avg_node_cost`, `max_node_cost`

**Categorical preprocessing:** most-frequent imputation and one-hot encoding for `workflow_type`, `workload_type`, `environment_condition`

**Excluded leakage columns:** `workflow_time`, `processing_time`, `communication_delay`, `memory_use`, `communication_fraction`, `sla_ok`, `predicted_workflow_time`, `predicted_sla_ok`

## SLA thresholds

- Global fallback: `256.864588`
- `cicd`: `407.726937`
- `iot`: `348.456815`
- `video`: `356.688553`

## Workflow-time regression

- MAE: `40.321953`
- RMSE: `53.075284`
- R2: `0.844369`
- Mean actual workflow time: `348.500896`
- Mean predicted workflow time: `348.339164`

## Actual vs Predicted Workflow-Time Distribution

| Statistic | Actual workflow_time | Predicted workflow_time | Difference |
|---|---:|---:|---:|
| Mean | 348.500896 | 348.339164 | -0.161732 |
| Median | 372.128865 | 394.361387 | 22.232522 |
| Standard deviation | 134.538493 | 123.181106 | -11.357386 |
| Minimum | 54.378285 | 94.223280 | 39.844995 |
| Maximum | 864.176953 | 585.883733 | -278.293221 |
| 25th percentile | 220.173644 | 211.779683 | -8.393962 |
| 75th percentile | 445.042471 | 437.108675 | -7.933796 |

- Mean error (predicted - actual): `-0.161732`
- Median error (predicted - actual): `22.232522`
- Mean percent error: `-0.046408%`
- Median percent error: `5.974415%`

The RF model shows distribution mismatch, so individual predictions or scheduler selection may be unreliable.

> **Warning:** Matching the mean and median does not prove the scheduler selects the best placement. It only shows that the prediction distribution is close to the actual validation distribution.

## SLA classification

- Accuracy: `0.821989`
- Precision: `0.849879`
- Recall (`sla_ok=1`): `0.762175`
- F1-score: `0.803641`
- SLA-violation recall (`sla_ok=0`): `0.876748`
- Validation SLA-positive samples: `41581`
- Validation SLA-negative samples: `45419`
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[39821, 5598], [9889, 31692]]`

## Representative validation cases

### Video

- Actual workflow time: `393.601789`
- Predicted workflow time: `394.602399`
- Absolute error: `1.000609`
- Actual `sla_ok`: `0`
- Predicted `sla_ok`: `0`
- Predicted SLA probability: `0.286938`
- Input features: `avg_node_cost=None`, `colocated_edges=0.0`, `communication_scale=1.3`, `dag_crossings=7.0`, `dag_depth=6.0`, `distinct_nodes_used=8`, `environment_condition=normal`, `max_node_cost=None`, `memory_scale=1.3`, `node_count=250`, `num_edges=7.0`, `num_tasks=8.0`, `workflow_type=video`, `workload_type=mixed`

### IoT

- Actual workflow time: `379.761719`
- Predicted workflow time: `381.439910`
- Absolute error: `1.678191`
- Actual `sla_ok`: `0`
- Predicted `sla_ok`: `0`
- Predicted SLA probability: `0.302160`
- Input features: `avg_node_cost=None`, `colocated_edges=0.0`, `communication_scale=1.3`, `dag_crossings=8.0`, `dag_depth=7.0`, `distinct_nodes_used=9`, `environment_condition=normal`, `max_node_cost=None`, `memory_scale=1.3`, `node_count=250`, `num_edges=8.0`, `num_tasks=9.0`, `workflow_type=iot`, `workload_type=mixed`

### CI/CD

- Actual workflow time: `447.411282`
- Predicted workflow time: `448.324769`
- Absolute error: `0.913486`
- Actual `sla_ok`: `0`
- Predicted `sla_ok`: `0`
- Predicted SLA probability: `0.273011`
- Input features: `avg_node_cost=None`, `colocated_edges=0.0`, `communication_scale=1.3`, `dag_crossings=8.0`, `dag_depth=8.0`, `distinct_nodes_used=9`, `environment_condition=normal`, `max_node_cost=None`, `memory_scale=1.3`, `node_count=250`, `num_edges=8.0`, `num_tasks=9.0`, `workflow_type=cicd`, `workload_type=mixed`

## Honest interpretation

- R2 is relatively strong on this validation split, though robustness still requires scenario- or source-grouped validation.
- Validation is encouraging, but this random row split alone is not enough to claim robust scheduler performance.
- This is model validation only. It does not train the final scheduler or compare ML scheduling against the regular scheduler.

## Run

```bash
python scripts/validate_rf_prediction_and_sla.py
```
