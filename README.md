# ML-based Cloud Workflow Scheduling

This repository contains the implementation and experimental results for Anwesha Ghosh's Master's thesis:

**ML-based Network and Resource-Aware Scheduling of Dockerized Applications on Kubernetes in Distributed Cloud Systems**

The current implementation is a Python simulation and machine learning evaluation framework. It investigates how resource conditions, network conditions, workflow structure, and task placement affect workflow execution time. Kubernetes and Docker are the intended deployment context; live Kubernetes scheduling is not implemented or evaluated here yet.

## Research tasks

1. **Prepare infrastructure and workload data.** Generate heterogeneous nodes and map workload samples to snapshots using CPU, memory, latency, bandwidth, packet loss, and availability. The real-data converter uses machine-usage CPU and memory samples and adds synthetic network and availability values.
2. **Model application workflows.** Define video processing, IoT, and CI/CD applications as task components and directed acyclic graphs (DAGs).
3. **Generate scheduling scenarios.** Place tasks on nodes and calculate processing time, communication delay, and total simulated workflow time across cluster sizes, workload profiles, and environment conditions.
4. **Audit and extract ML features.** Identify missing data, conflicting workflow metadata, and target leakage; extract workflow and placement features with source-file provenance.
5. **Train a workflow-time predictor.** Fit a Random Forest regression pipeline and evaluate it using a split grouped by source file.
6. **Validate predictions and SLA classification.** Run a separate experiment to predict workflow time and classify compliance with training-derived SLA thresholds.
7. **Compare scheduling decisions.** Select candidate placements by predicted workflow time, then compare simulated execution time against the default random placement baseline under the same node snapshot.
8. **Analyse results.** Generate statistical summaries, feature importance plots, prediction diagnostics, and representative best, worst, and tied scheduling cases.

## Experimental configurations

The default scheduler comparison evaluates 300 configurations:

| Dimension | Values |
|---|---|
| Application | Video, IoT, CI/CD |
| Node count | 15, 30, 60, 120, 250 |
| Workload | Light, heavy, data intensive, mixed |
| Environment | Normal, fast network, slow network, light memory use, heavy memory use |
| Candidate placements | 100 per scenario by default |
| Random seed | 42 by default |

The simulator sums component processing costs and communication costs between tasks placed on different nodes. This is a simplified cost model rather than a parallel DAG makespan calculation or measured container execution time.

## Repository structure

| Path | Purpose |
|---|---|
| `simulation/` | Infrastructure, node mapping, workflow definitions, scenario generation, datasets, and scenario analysis |
| `results_video/`, `results_iot/`, `results_cicd/` | Workflow-specific definitions and generated scenario CSVs |
| `machine_usage_days_1_to_8_grouped_300_seconds.csv/` | Machine-usage dataset and conversion script |
| `scripts/audit_and_extract_features.py` | Dataset audit and feature extraction |
| `ml_model/train_rf.py` | Current Random Forest regression training pipeline |
| `scripts/validate_rf_prediction_and_sla.py` | Separate regression and SLA classification validation |
| `scripts/compare_regular_vs_ml_scheduler.py` | Simulation comparison and optional offline candidate re-ranking |
| `scripts/compare_regular_vs_ml_cases.py` | Case-level comparison analysis |
| `scheduler/ml_scheduler.py` | Earlier scheduler prototype using `ml_model/rf_bundle.joblib` |
| `Analysis/` | Workflow, scenario, statistical, and dynamic comparison plots |
| `outputs/` | Features, trained pipeline, metrics, predictions, reports, and figures |

Root-level simulation scripts and datasets are retained from earlier experiments. Use the following pipeline for the current training and scheduler comparison.

## Setup

Use Python 3.10 or later and a virtual environment. Dependencies are not currently pinned; scikit-learn 1.2 or later is required by the training script's encoder API.

```bash
git clone https://github.com/Anwesha0996/ml_cloud_workflow_scheduling_Master_Thesis.git
cd ml_cloud_workflow_scheduling_Master_Thesis
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate on Linux or macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install numpy pandas matplotlib joblib "scikit-learn>=1.2"
```

If an included Joblib model is incompatible with the installed scikit-learn version, retrain it using the pipeline below.

## Run the current pipeline

Run these commands from the repository root. Included scenario datasets and simulation inputs allow the pipeline to run without first regenerating infrastructure. Commands can replace existing output artifacts. Training and validation process hundreds of thousands of rows and can take time and memory.

### 1. Audit datasets and extract features

```bash
python scripts/audit_and_extract_features.py
```

Writes `outputs/feature_audit/`, including `extracted_ml_dataset.csv`, file-level audits, feature availability, and metadata-conflict findings.

### 2. Train the regression baseline

```bash
python ml_model/train_rf.py
```

Reads the extracted dataset and writes `outputs/rf_regression/`, including `rf_regression_pipeline.joblib`, test predictions, metrics, and feature importance plots.

The 12 inputs are `workflow_type`, `workload_type`, `environment_condition`, `node_count`, `communication_scale`, `memory_scale`, `num_tasks`, `num_edges`, `dag_depth`, `distinct_nodes_used`, `dag_crossings`, and `colocated_edges`. The target is `workflow_time`. Source-file identifiers are used for splitting and traceability, not as model inputs. Rows missing required inputs or the target are dropped.

### 3. Compare regular and ML-selected scheduling

```bash
python scripts/compare_regular_vs_ml_scheduler.py --mode simulation
```

The script compares a random baseline placement with a placement selected from random candidates by the trained model. Both use the same simulator and node snapshot. Predicted time selects the candidate; simulated time determines the outcome.

Writes `outputs/ml_scheduler_comparison/`, including `comparison_results.csv`, `summary_metrics.json`, `summary_report.md`, and plots.

For a smaller run saved separately:

```bash
python scripts/compare_regular_vs_ml_scheduler.py --mode simulation --workflows video --workloads light --environments normal --node-counts 30 --candidates 20 --output-dir outputs/quick_comparison
```

### 4. Analyse individual scheduling cases

```bash
python scripts/compare_regular_vs_ml_cases.py
```

Writes representative best, worst, and tied cases to `outputs/case_comparison/` using the default comparison results.

### 5. Run separate prediction and SLA validation

```bash
python scripts/validate_rf_prediction_and_sla.py
```

Writes regression metrics, SLA classification metrics, a confusion matrix, plots, and examples to `outputs/ml_validation/`. This experiment uses separate models and a random row split; it does not train the pipeline used in step 3.

## Included experimental results

These values describe the committed reports. The experiments were not rerun for this documentation update.

### Grouped Random Forest regression

The [training report](outputs/rf_regression/training_report.md) records an 80/20 split grouped by source file: 219,000 training rows and 57,000 test rows across 73 training and 19 test source files, with no source-file group overlap.

| Metric | Value |
|---|---:|
| MAE | 45.3571 |
| RMSE | 58.4865 |
| R-squared | 0.4561 |

The report records 159,000 rows dropped for missing values and 216,000 exact duplicate feature/target rows. Grouping by file does not guarantee independence of related or duplicated observations across different files.

### Scheduler comparison

The [comparison report](outputs/ml_scheduler_comparison/summary_report.md) records:

| Measure | Value |
|---|---:|
| Scenarios | 300 |
| Mean baseline simulated workflow time | 302.4837 |
| Mean ML-selected simulated workflow time | 302.8250 |
| ML wins / baseline wins / ties | 142 / 125 / 33 |
| Mean per-scenario improvement | -0.9337% |
| Median per-scenario improvement | 0.0000% |

Improvement is `100 * (regular_actual_time - ml_selected_actual_time) / regular_actual_time`; positive values mean faster execution. The mean percentage averages individual scenario percentages and differs from the percentage change between the two overall means.

**The current ML selection method does not improve overall average simulated workflow time over the random baseline.** Outcomes vary by workload and scenario. The baseline is the repository's random placement generator, not the Kubernetes default scheduler. Times are simulator outputs, not measured Kubernetes execution times.

### Separate prediction and SLA validation

The [validation report](outputs/ml_validation/rf_validation_report.md) records regression R-squared of 0.8444 and SLA classification accuracy of 0.8220. It uses a random row split and a different feature/preprocessing setup, including imputed `avg_node_cost` and `max_node_cost`. Related observations may occur in both subsets, so these results may be optimistic and are not directly comparable with grouped regression results.

## Current limitations and next steps

- Validate scheduling on a real Kubernetes cluster with containerized workflows.
- Strengthen evaluation with independent scenarios, duplicate handling, and repeated seeds; resolve workflow metadata conflicts and incomplete feature coverage.
- Improve placement-sensitive prediction and candidate search, then evaluate actual scheduling outcomes across workloads.
- Evaluate SLA compliance of selected schedules and compare against additional scheduling baselines.
- Pin dependencies and reconcile older model artifacts with the current pipeline. `scheduler/ml_scheduler.py` expects `rf_bundle.joblib`, while the current trainer produces `outputs/rf_regression/rf_regression_pipeline.joblib`.
- Review paths before regenerating data: the real-data converter has a machine-specific input path, `simulation/node_mapper.py` expects to run inside `simulation/`, and other generators use paths relative to the project root.

## Author

**Anwesha Ghosh**
MSc Information and Communication Engineering
AAU Klagenfurt / Tu Dresden
