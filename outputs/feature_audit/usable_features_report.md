# Feature Audit and Extraction Report

## Run the pipeline

From the project root:

```bash
python scripts/audit_and_extract_features.py
```

(`python3 scripts/audit_and_extract_features.py` is equivalent on systems where `python` is not configured.)

## Scan summary

- Total files discovered: 145
- Files read successfully: 145
- Total rows scanned: 435000
- Extracted dataset rows: 435000
- Read errors: 0

The extracted dataset contains one row per successfully read source row. DAG depth is measured as the longest source-to-sink path in edges.

`source_file`, `scenario_id`, `scheduler_type`, and `candidate_id` are provenance identifiers only. They are not recommended as model inputs.

## Candidate feature availability and variation

| Feature | Non-missing rows | Available % | Unique values | Assessment |
|---|---:|---:|---:|---|
| `workflow_type` | 309000 | 71.03% | 3 | Usable variation |
| `node_count` | 435000 | 100.00% | 12 | Usable variation |
| `workload_type` | 351000 | 80.69% | 4 | Usable variation |
| `environment_condition` | 411000 | 94.48% | 5 | Usable variation |
| `communication_scale` | 411000 | 94.48% | 7 | Usable variation |
| `memory_scale` | 411000 | 94.48% | 7 | Usable variation |
| `num_tasks` | 309000 | 71.03% | 2 | Usable variation |
| `num_edges` | 309000 | 71.03% | 2 | Usable variation |
| `dag_depth` | 309000 | 71.03% | 3 | Usable variation |
| `distinct_nodes_used` | 435000 | 100.00% | 8 | Usable variation |
| `dag_crossings` | 300000 | 68.97% | 6 | Usable variation |
| `colocated_edges` | 300000 | 68.97% | 5 | Usable variation |
| `avg_node_cost` | 24000 | 5.52% | 3000 | Missing often |
| `max_node_cost` | 24000 | 5.52% | 110 | Missing often |

## Availability findings

**Available features:** `workflow_type`, `node_count`, `workload_type`, `environment_condition`, `communication_scale`, `memory_scale`, `num_tasks`, `num_edges`, `dag_depth`, `distinct_nodes_used`, `dag_crossings`, `colocated_edges`, `avg_node_cost`, `max_node_cost`

**Missing often (more than 50% missing):** `avg_node_cost`, `max_node_cost`

**Low variation (zero or one observed value):** None

## Target leakage

The following columns are unsafe as model inputs when predicting `workflow_time`:

- `communication_delay`
- `communication_fraction`
- `memory_use`
- `processing_time`
- `workflow_time`

`workflow_time` is retained only as the prediction target. The other leakage columns are excluded from `extracted_ml_dataset.csv`.

## Recommended safe Random Forest features

- `workflow_type`
- `node_count`
- `workload_type`
- `environment_condition`
- `communication_scale`
- `memory_scale`
- `num_tasks`
- `num_edges`
- `dag_depth`
- `distinct_nodes_used`
- `dag_crossings`
- `colocated_edges`

These recommendations exclude target leakage and require at least 50% availability plus more than one observed value. Categorical features must be encoded before fitting a scikit-learn Random Forest.

Placement features are leakage-safe with respect to measured runtime, but they are only valid predictors when a candidate placement is known at prediction time.

## Placement feature coverage

- `distinct_nodes_used` is available in 145 of 145 successfully read files.
- `dag_crossings` is available in 100 of 145 successfully read files.
- `colocated_edges` is available in 100 of 145 successfully read files.

> **Warning:** Placement features exist only in a subset of files. Training directly on all rows will require filtering, imputation, or separate model variants.

## Metadata consistency

- Files whose filename/path workflow conflicts with assignment task columns: 11

> **Warning:** Conflicting files keep the filename/path `workflow_type`, but DAG placement metrics are left missing rather than being calculated against the wrong workflow. See `workflow_metadata_conflict` in `file_level_audit.csv`.

## Output files

- `outputs/feature_audit/feature_column_audit.csv`
- `outputs/feature_audit/file_level_audit.csv`
- `outputs/feature_audit/extracted_ml_dataset.csv`
- `outputs/feature_audit/usable_features_report.md`
