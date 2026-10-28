# simulation/workflow_profiles.py

def get_workflow_profile(name: str) -> dict:
    """
    Returns scaling values that define a workflow type.

    Each workflow type controls:
    - runtime_scale: affects how long tasks take
    - comm_scale: affects communication delay between components
    - memory_scale: affects how much memory impacts performance
    """
    name = (name or "baseline").strip().lower()

    profiles = {
        # default workflow
        "baseline": {
            "runtime_scale": 1.0,
            "comm_scale": 1.0,
            "memory_scale": 1.0
        },

        # light workload (faster tasks)
        "light_workload": {
            "runtime_scale": 0.7,
            "comm_scale": 1.0,
            "memory_scale": 1.0
        },

        # heavy workload (longer processing time)
        "heavy_workload": {
            "runtime_scale": 1.5,
            "comm_scale": 1.0,
            "memory_scale": 1.0
        },

        # data-intensive workload (more memory + communication impact)
        "data_intensive_workload": {
            "runtime_scale": 1.1,
            "comm_scale": 1.6,
            "memory_scale": 2.0
        },

        # mixed workload (balanced case)
        "mixed_workload": {
            "runtime_scale": 1.2,
            "comm_scale": 1.3,
            "memory_scale": 1.3
        },
    }

    if name not in profiles:
        raise ValueError(
            f"Unknown workflow type '{name}'. Choose one of: {list(profiles.keys())}"
        )

    return profiles[name]