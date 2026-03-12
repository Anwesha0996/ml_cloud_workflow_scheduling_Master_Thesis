# simulation/workflow_profiles.py

def get_workflow_profile(name: str) -> dict:
    """
    Returns scaling factors that define a workflow "type".
    You can add more profiles later.

    Keys:
      runtime_scale: multiplies component runtimes (short vs long)
      comm_scale: multiplies communication penalty (network-heavy)
      memory_scale: multiplies the memory contribution in runtime (storage/data-heavy)
    """
    name = (name or "baseline").strip().lower()

    profiles = {
        # normal workflow (your current behavior)
        "baseline":   {"runtime_scale": 1.0, "comm_scale": 1.0, "memory_scale": 1.0},

        # shorter tasks overall
        "short":      {"runtime_scale": 0.7, "comm_scale": 1.0, "memory_scale": 1.0},

        # longer tasks overall
        "long":       {"runtime_scale": 1.5, "comm_scale": 1.0, "memory_scale": 1.0},

        # data-heavy workflow: memory matters more + network penalty higher
        "dataheavy":  {"runtime_scale": 1.1, "comm_scale": 1.6, "memory_scale": 2.0},

        # optional mixed workflow (if you want a 4th)
        "mixed":      {"runtime_scale": 1.2, "comm_scale": 1.3, "memory_scale": 1.3},
    }

    if name not in profiles:
        raise ValueError(f"Unknown workflow profile '{name}'. Choose one of: {list(profiles.keys())}")

    return profiles[name]
