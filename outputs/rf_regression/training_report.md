# Random Forest Regression Baseline

## Split strategy

- Strategy: grouped 80/20 split by source_file
- Training rows: 219000
- Test rows: 57000
- Complete rows used: 276000
- Rows dropped for missing values: 159000
- Exact duplicate feature/target rows: 216000

- Training source-file groups: 73
- Test source-file groups: 19
- Source-file group overlap: 0

Source-file identifiers are used only for splitting and traceability, never as model inputs.

## Test metrics

- MAE: 45.357059
- RMSE: 58.486526
- R2: 0.456097

## Aggregate feature importance

| Feature | Importance |
|---|---:|
| `workload_type` | 0.543732 |
| `environment_condition` | 0.159573 |
| `memory_scale` | 0.107094 |
| `dag_depth` | 0.082767 |
| `workflow_type` | 0.069851 |
| `communication_scale` | 0.028941 |
| `node_count` | 0.005057 |
| `distinct_nodes_used` | 0.000889 |
| `num_edges` | 0.000601 |
| `num_tasks` | 0.000527 |
| `dag_crossings` | 0.000495 |
| `colocated_edges` | 0.000474 |

The model uses only the 12 requested leakage-safe features. `avg_node_cost` and `max_node_cost` are excluded.

## Artifacts

- `rf_regression_pipeline.joblib`
- `metrics.json`
- `test_predictions.csv`
- `feature_importance.csv`
- `encoded_feature_importance.csv`
- `feature_importance.png`
- `predicted_vs_actual.png`
- `residuals_vs_predicted.png`
