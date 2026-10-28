# Regular vs ML Scheduler Comparison

- Comparison possible: Yes
- Evaluation type: **true simulation-based comparison**
- Scenarios compared: 300
- Mean regular actual time: 302.483667
- Mean ML-selected actual time: 302.824965
- Mean percent improvement: -0.933718%
- Median percent improvement: 0.000000%
- ML wins: 142
- Regular scheduler wins: 125
- Ties: 33

The comparison is regular scheduler actual/simulated `workflow_time` versus ML-selected schedule actual/simulated `workflow_time`. Predicted time is used only to select a candidate, never as the final comparison outcome.

## Main limitation

This is a deterministic simulator comparison, not a live Kubernetes deployment. The regular scheduler is the repository's existing default random placement generator, and the ML scheduler chooses from a finite random candidate pool under the same simulated node snapshot.

## By workload type

| workload_type | Scenarios | Regular mean | ML mean | Mean improvement % | ML wins | Regular wins | Ties |
|---|---:|---:|---:|---:|---:|---:|---:|
| data_intensive | 75 | 348.0477 | 352.1099 | -1.6261 | 36 | 31 | 8 |
| heavy | 75 | 355.0407 | 349.4970 | 0.4052 | 39 | 29 | 7 |
| light | 75 | 185.8147 | 181.0628 | 1.3880 | 42 | 28 | 5 |
| mixed | 75 | 321.0316 | 328.6302 | -3.9020 | 25 | 37 | 13 |

## By node count

| node_count | Scenarios | Regular mean | ML mean | Mean improvement % | ML wins | Regular wins | Ties |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15 | 60 | 267.3086 | 265.1319 | -1.0591 | 25 | 21 | 14 |
| 30 | 60 | 267.5439 | 269.0932 | -0.6347 | 32 | 24 | 4 |
| 60 | 60 | 271.4578 | 270.7011 | -0.5729 | 29 | 28 | 3 |
| 120 | 60 | 299.4819 | 297.3721 | -0.5780 | 27 | 30 | 3 |
| 250 | 60 | 406.6261 | 411.8265 | -1.8239 | 29 | 22 | 9 |

## By environment condition

| environment_condition | Scenarios | Regular mean | ML mean | Mean improvement % | ML wins | Regular wins | Ties |
|---|---:|---:|---:|---:|---:|---:|---:|
| fast_network | 60 | 273.1617 | 276.7119 | -2.2910 | 24 | 29 | 7 |
| heavy_memory_use | 60 | 357.9861 | 358.3939 | -0.4730 | 28 | 23 | 9 |
| light_memory_use | 60 | 267.5718 | 267.8376 | -0.7605 | 30 | 24 | 6 |
| normal | 60 | 288.6924 | 284.7628 | -0.3756 | 30 | 26 | 4 |
| slow_network | 60 | 325.0063 | 326.4186 | -0.7685 | 30 | 23 | 7 |
