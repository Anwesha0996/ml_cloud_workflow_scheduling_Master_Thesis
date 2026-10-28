# Regular vs ML Case-Level Comparison

This is a simulation-based comparison, not a real Kubernetes cluster result.

- Input: `outputs/ml_scheduler_comparison/comparison_results.csv`
- Total cases: 300

> **Warning:** `ml_predicted_time_for_selected_candidate` is not the final result. It is only the value used by the ML model to select a candidate. The final comparison must use `ml_selected_actual_time`.

## Representative Cases

Each representative is the case closest to its workflow's median percent improvement, avoiding selection of an unusually good or bad case.

### Video

- Scenario: `video-heavy-heavy_memory_use-nodes60-rep0`
- Workflow type: `video`
- Workload type: `heavy`
- Environment condition: `heavy_memory_use`
- Node count: `60`
- Regular actual time: `355.681999`
- ML predicted time for selected candidate: `424.445890`
- ML-selected actual time: `348.289857`
- Absolute improvement: `7.392142`
- Percent improvement: `2.078301%`
- Winner: `ml`
- Number of candidates: `100`
- Workflow median percent improvement: `2.184029%`
- Interpretation: ML selected a better simulated placement.

### IoT

- Scenario: `iot-light-heavy_memory_use-nodes30-rep0`
- Workflow type: `iot`
- Workload type: `light`
- Environment condition: `heavy_memory_use`
- Node count: `30`
- Regular actual time: `180.915991`
- ML predicted time for selected candidate: `205.851171`
- ML-selected actual time: `181.396701`
- Absolute improvement: `-0.480710`
- Percent improvement: `-0.265709%`
- Winner: `regular`
- Number of candidates: `100`
- Workflow median percent improvement: `-0.295048%`
- Interpretation: The regular simulated scheduler was better.

### CI/CD

- Scenario: `cicd-data_intensive-fast_network-nodes15-rep0`
- Workflow type: `cicd`
- Workload type: `data_intensive`
- Environment condition: `fast_network`
- Node count: `15`
- Regular actual time: `321.418347`
- ML predicted time for selected candidate: `451.996399`
- ML-selected actual time: `321.418347`
- Absolute improvement: `0.000000`
- Percent improvement: `0.000000%`
- Winner: `tie`
- Number of candidates: `100`
- Workflow median percent improvement: `0.000000%`
- Interpretation: Both produced equal simulated workflow time.

## Best, Worst, and Nearest-Zero Cases

| case_type | workflow_type | scenario_id | regular_actual_time | ml_selected_actual_time | percent_improvement | winner |
|---|---:|---:|---:|---:|---:|---:|
| best_ml | video | video-light-fast_network-nodes120-rep0 | 190.396924 | 112.033033 | 41.158171 | ml |
| worst_ml | video | video-mixed-heavy_memory_use-nodes30-rep0 | 247.074620 | 381.777843 | -54.519247 | regular |
| tie_or_nearest_zero | video | video-heavy-heavy_memory_use-nodes30-rep0 | 287.868219 | 287.868219 | 0.000000 | tie |
| best_ml | iot | iot-heavy-normal-nodes250-rep0 | 526.710557 | 283.388448 | 46.196551 | ml |
| worst_ml | iot | iot-light-light_memory_use-nodes15-rep0 | 111.420139 | 161.821047 | -45.235007 | regular |
| tie_or_nearest_zero | iot | iot-data_intensive-heavy_memory_use-nodes30-rep0 | 404.377089 | 404.377089 | 0.000000 | tie |
| best_ml | cicd | cicd-heavy-fast_network-nodes120-rep0 | 391.012984 | 283.114259 | 27.594666 | ml |
| worst_ml | cicd | cicd-data_intensive-fast_network-nodes250-rep0 | 410.805802 | 652.603064 | -58.859262 | regular |
| tie_or_nearest_zero | cicd | cicd-data_intensive-fast_network-nodes15-rep0 | 321.418347 | 321.418347 | 0.000000 | tie |

## Mean Result by Workflow Type

| workflow_type | scenario_count | mean_regular_actual_time | mean_ml_selected_actual_time | mean_absolute_improvement | mean_percent_improvement |
|---|---:|---:|---:|---:|---:|
| cicd | 100 | 337.469331 | 334.381103 | 3.088229 | 0.395889 |
| iot | 100 | 279.309428 | 281.595032 | -2.285604 | -2.589766 |
| video | 100 | 290.672242 | 292.498761 | -1.826519 | -0.607276 |

## Mean Result by Workload Type

| workload_type | scenario_count | mean_regular_actual_time | mean_ml_selected_actual_time | mean_absolute_improvement | mean_percent_improvement |
|---|---:|---:|---:|---:|---:|
| data_intensive | 75 | 348.047677 | 352.109882 | -4.062205 | -1.626075 |
| heavy | 75 | 355.040719 | 349.496982 | 5.543737 | 0.405187 |
| light | 75 | 185.814720 | 181.062829 | 4.751891 | 1.388029 |
| mixed | 75 | 321.031553 | 328.630168 | -7.598615 | -3.902011 |

## Mean Result by Node Count

| node_count | scenario_count | mean_regular_actual_time | mean_ml_selected_actual_time | mean_absolute_improvement | mean_percent_improvement |
|---|---:|---:|---:|---:|---:|
| 15 | 60 | 267.308606 | 265.131906 | 2.176700 | -1.059134 |
| 30 | 60 | 267.543893 | 269.093249 | -1.549356 | -0.634663 |
| 60 | 60 | 271.457805 | 270.701149 | 0.756655 | -0.572927 |
| 120 | 60 | 299.481935 | 297.372061 | 2.109874 | -0.578005 |
| 250 | 60 | 406.626097 | 411.826461 | -5.200364 | -1.823858 |

## Win/Tie/Loss Counts

| winner | count |
|---|---:|
| ml | 142 |
| regular | 125 |
| tie | 33 |

## Run

```bash
python scripts/compare_regular_vs_ml_cases.py
```
