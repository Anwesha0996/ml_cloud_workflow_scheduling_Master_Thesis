# simulation/sc3_conditions.py

def get_condition(name: str) -> dict:
    """
    Returns scaling values for different environment conditions.

    Each condition controls:
    - comm_scale: affects communication delay
    - memory_scale: affects memory impact
    """
    name = (name or "normal").strip().lower()

    conditions = {
        "normal": {
            "comm_scale": 1.0,
            "memory_scale": 1.0
        },

        "fast_network": {
            "comm_scale": 0.6,
            "memory_scale": 1.0
        },

        "slow_network": {
            "comm_scale": 2.0,
            "memory_scale": 1.0
        },

        "light_memory_use": {
            "comm_scale": 1.0,
            "memory_scale": 0.7
        },

        "heavy_memory_use": {
            "comm_scale": 1.0,
            "memory_scale": 2.0
        },
    }

    if name not in conditions:
        raise ValueError(
            f"Unknown environment condition '{name}'. Choose one of: {list(conditions.keys())}"
        )

    return conditions[name]