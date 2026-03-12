# simulation/sc3_conditions.py

def get_condition(name: str) -> dict:
    """
    SC3: "environment conditions" (network/storage intensity)
    """
    name = (name or "normal").strip().lower()

    conditions = {
        "normal":        {"comm_scale": 1.0, "memory_scale": 1.0},
        "network_good":  {"comm_scale": 0.6, "memory_scale": 1.0},
        "network_bad":   {"comm_scale": 2.0, "memory_scale": 1.0},
        "storage_light": {"comm_scale": 1.0, "memory_scale": 0.7},
        "storage_heavy": {"comm_scale": 1.0, "memory_scale": 2.0},
    }

    if name not in conditions:
        raise ValueError(f"Unknown SC3 condition '{name}'. Choose: {list(conditions.keys())}")
    return conditions[name]
