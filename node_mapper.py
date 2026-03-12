# simulation/node_mapper.py
import os
import numpy as np
import pandas as pd


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def create_nodes(n_nodes: int, seed: int = 42) -> pd.DataFrame:
    """
    Create a heterogeneous node inventory.
    Scales cleanly to 50 / 100 / 150 / 250 nodes.
    """
    rng = np.random.default_rng(seed)

    # proportions: 30% high, 40% mid, 30% low
    n_high = int(0.3 * n_nodes)
    n_mid  = int(0.4 * n_nodes)
    n_low  = n_nodes - n_high - n_mid

    tiers = (["high"] * n_high) + (["mid"] * n_mid) + (["low"] * n_low)

    node_rows = []

    for i, tier in enumerate(tiers):
        if tier == "high":
            cpu_capacity = int(rng.integers(64, 97))
            mem_capacity = int(rng.integers(64000, 128001))
            base_latency = int(rng.integers(5, 25))
            base_bandwidth = int(rng.integers(1200, 4001))
            base_packet_loss = float(rng.uniform(0.0, 0.5))
            base_availability = float(rng.uniform(0.98, 0.999))

        elif tier == "mid":
            cpu_capacity = int(rng.integers(32, 65))
            mem_capacity = int(rng.integers(32000, 64001))
            base_latency = int(rng.integers(15, 60))
            base_bandwidth = int(rng.integers(600, 1801))
            base_packet_loss = float(rng.uniform(0.2, 1.0))
            base_availability = float(rng.uniform(0.965, 0.995))

        else:  # low tier
            cpu_capacity = int(rng.integers(16, 33))
            mem_capacity = int(rng.integers(16000, 32001))
            base_latency = int(rng.integers(40, 120))
            base_bandwidth = int(rng.integers(200, 1001))
            base_packet_loss = float(rng.uniform(0.8, 2.5))
            base_availability = float(rng.uniform(0.95, 0.99))

        rack = f"rack_{i % 4}"  # simple topology

        node_rows.append({
            "node_id": f"node_{i+1}",
            "tier": tier,
            "rack": rack,
            "cpu_capacity": cpu_capacity,
            "memory_capacity": mem_capacity,
            "base_latency": base_latency,
            "base_bandwidth": base_bandwidth,
            "base_packet_loss": base_packet_loss,
            "base_availability": base_availability
        })

    return pd.DataFrame(node_rows)


def compute_score_for_tier(row: pd.Series, tier: str) -> float:
    """
    Score how well a sample matches a node tier.
    Higher score => more suitable tier.
    """
    cpu = float(row["cpu"]) / 100.0
    mem = min(float(row["memory"]) / 16000.0, 1.0)
    lat = min(float(row["latency"]) / 300.0, 1.0)
    bw  = min(float(row["bandwidth"]) / 1000.0, 1.0)
    pl  = min(float(row["packet_loss"]) / 5.0, 1.0)
    av  = (float(row["availability"]) - 0.90) / (0.999 - 0.90)

    if tier == "high":
        return (
            2.0 * (1.0 - lat) +
            2.0 * bw +
            1.5 * (1.0 - pl) +
            1.5 * av +
            0.5 * (1.0 - cpu) +
            0.3 * mem
        )

    if tier == "mid":
        return (
            1.0 * (1.0 - abs(lat - 0.35)) +
            1.0 * (1.0 - abs(bw - 0.55)) +
            1.0 * (1.0 - abs(pl - 0.25)) +
            1.0 * (1.0 - abs(av - 0.70)) +
            0.7 * (1.0 - abs(cpu - 0.55)) +
            0.4 * (1.0 - abs(mem - 0.55))
        )

    if tier == "low":
        return (
            2.0 * lat +
            2.0 * (1.0 - bw) +
            1.5 * pl +
            1.5 * (1.0 - av) +
            0.6 * cpu +
            0.2 * (1.0 - mem)
        )

    return 0.0


def assign_nodes(df: pd.DataFrame, nodes: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Assign each time-series row to a node based on tier scoring.
    """
    rng = np.random.default_rng(seed)

    tier_to_nodes = {
        t: nodes[nodes["tier"] == t]["node_id"].tolist()
        for t in ["high", "mid", "low"]
    }

    chosen_nodes = []
    chosen_tiers = []

    for _, row in df.iterrows():
        scores = {
            "high": compute_score_for_tier(row, "high"),
            "mid":  compute_score_for_tier(row, "mid"),
            "low":  compute_score_for_tier(row, "low")
        }
        tier = max(scores, key=scores.get)
        node_id = rng.choice(tier_to_nodes[tier])

        chosen_tiers.append(tier)
        chosen_nodes.append(node_id)

    out = df.copy()
    out.insert(0, "t", np.arange(len(out)))
    out.insert(1, "node_id", chosen_nodes)
    out.insert(2, "node_tier", chosen_tiers)
    return out


def main() -> None:
    outdir = "simulation"
    ensure_outdir(outdir)

    # 🔧 CHANGE THIS FOR SC1 EXPERIMENTS
    N_NODES = int(os.environ.get("N_NODES", 250))

    in_csv = os.path.join(outdir, "data.csv")
    if not os.path.isfile(in_csv):
        raise FileNotFoundError("Run data_generator.py first")

    df = pd.read_csv(in_csv)

    nodes = create_nodes(n_nodes=N_NODES, seed=42)
    nodes.to_csv(os.path.join(outdir, "nodes.csv"), index=False)

    timeseries = assign_nodes(df, nodes, seed=42)
    timeseries.to_csv(os.path.join(outdir, "timeseries.csv"), index=False)

    print("✅ Node mapping completed")
    print(f"🔢 Nodes used: {N_NODES}")
    print(timeseries["node_tier"].value_counts())


if __name__ == "__main__":
    main()
