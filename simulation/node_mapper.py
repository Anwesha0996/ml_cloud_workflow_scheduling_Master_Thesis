import os
import numpy as np
import pandas as pd


def ensure_output_folder(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def create_nodes(number_of_nodes: int, seed: int = 42) -> pd.DataFrame:
    """
    Create a mixed set of computing nodes.
    """
    rng = np.random.default_rng(seed)

    # 30% powerful, 40% standard, 30% limited
    n_powerful = int(0.3 * number_of_nodes)
    n_standard = int(0.4 * number_of_nodes)
    n_limited = number_of_nodes - n_powerful - n_standard

    node_types = (
        ["powerful"] * n_powerful
        + ["standard"] * n_standard
        + ["limited"] * n_limited
    )

    node_rows = []

    for i, node_type in enumerate(node_types):
        if node_type == "powerful":
            cpu_capacity = int(rng.integers(64, 97))
            memory_capacity = int(rng.integers(64000, 128001))
            base_latency = int(rng.integers(5, 25))
            base_bandwidth = int(rng.integers(1200, 4001))
            base_packet_loss = float(rng.uniform(0.0, 0.5))
            base_availability = float(rng.uniform(0.98, 0.999))

        elif node_type == "standard":
            cpu_capacity = int(rng.integers(32, 65))
            memory_capacity = int(rng.integers(32000, 64001))
            base_latency = int(rng.integers(15, 60))
            base_bandwidth = int(rng.integers(600, 1801))
            base_packet_loss = float(rng.uniform(0.2, 1.0))
            base_availability = float(rng.uniform(0.965, 0.995))

        else:  # limited
            cpu_capacity = int(rng.integers(16, 33))
            memory_capacity = int(rng.integers(16000, 32001))
            base_latency = int(rng.integers(40, 120))
            base_bandwidth = int(rng.integers(200, 1001))
            base_packet_loss = float(rng.uniform(0.8, 2.5))
            base_availability = float(rng.uniform(0.95, 0.99))

        rack_id = f"rack_{i % 4}"

        node_rows.append({
            "node_id": f"node_{i+1}",
            "node_type": node_type,
            "rack": rack_id,
            "cpu_capacity": cpu_capacity,
            "memory_capacity": memory_capacity,
            "base_latency": base_latency,
            "base_bandwidth": base_bandwidth,
            "base_packet_loss": base_packet_loss,
            "base_availability": base_availability
        })

    return pd.DataFrame(node_rows)


def compute_match_score(row: pd.Series, node_type: str) -> float:
    """
    Compute how well one workload sample matches one node type.
    Higher score means a better match.
    """
    cpu = float(row["cpu"]) / 100.0
    memory = min(float(row["memory"]) / 16000.0, 1.0)
    latency = min(float(row["latency"]) / 300.0, 1.0)
    bandwidth = min(float(row["bandwidth"]) / 1000.0, 1.0)
    packet_loss = min(float(row["packet_loss"]) / 5.0, 1.0)
    availability = (float(row["availability"]) - 0.90) / (0.999 - 0.90)

    if node_type == "powerful":
        return (
            2.0 * (1.0 - latency) +
            2.0 * bandwidth +
            1.5 * (1.0 - packet_loss) +
            1.5 * availability +
            0.5 * (1.0 - cpu) +
            0.3 * memory
        )

    elif node_type == "standard":
        return (
            1.0 * (1.0 - abs(latency - 0.35)) +
            1.0 * (1.0 - abs(bandwidth - 0.55)) +
            1.0 * (1.0 - abs(packet_loss - 0.25)) +
            1.0 * (1.0 - abs(availability - 0.70)) +
            0.7 * (1.0 - abs(cpu - 0.55)) +
            0.4 * (1.0 - abs(memory - 0.55))
        )

    else:  # limited
        return (
            2.0 * latency +
            2.0 * (1.0 - bandwidth) +
            1.5 * packet_loss +
            1.5 * (1.0 - availability) +
            0.6 * cpu +
            0.2 * (1.0 - memory)
        )


def assign_nodes_to_samples(
    workload_data: pd.DataFrame,
    nodes: pd.DataFrame,
    seed: int = 42
) -> pd.DataFrame:
    """
    Assign each workload sample to a node based on best node type match.
    """
    rng = np.random.default_rng(seed)

    node_type_to_ids = {
        node_type: nodes[nodes["node_type"] == node_type]["node_id"].tolist()
        for node_type in ["powerful", "standard", "limited"]
    }

    chosen_node_ids = []
    chosen_node_types = []

    for _, row in workload_data.iterrows():
        scores = {
            "powerful": compute_match_score(row, "powerful"),
            "standard": compute_match_score(row, "standard"),
            "limited": compute_match_score(row, "limited")
        }

        best_node_type = max(scores, key=scores.get)
        selected_node_id = rng.choice(node_type_to_ids[best_node_type])

        chosen_node_types.append(best_node_type)
        chosen_node_ids.append(selected_node_id)

    output_data = workload_data.copy()
    output_data.insert(0, "time_index", np.arange(len(output_data)))
    output_data.insert(1, "node_id", chosen_node_ids)
    output_data.insert(2, "node_type", chosen_node_types)

    return output_data


def main() -> None:
    # Since you run inside the simulation folder:
    output_folder = "."
    ensure_output_folder(output_folder)

    # Change this with: N_NODES=100 python node_mapper.py
    number_of_nodes = int(os.environ.get("N_NODES", 250))

    # Since you copied data_real.csv -> data.csv, read data.csv
    input_csv = os.path.join(output_folder, "data.csv")
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(
            f"Missing input file: {input_csv}\n"
            "First create data_real.csv and copy it to data.csv."
        )

    workload_data = pd.read_csv(input_csv)

    required_columns = ["cpu", "memory", "latency", "bandwidth", "packet_loss", "availability"]
    missing_columns = [col for col in required_columns if col not in workload_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {input_csv}: {missing_columns}")

    nodes = create_nodes(number_of_nodes=number_of_nodes, seed=42)
    nodes.to_csv(os.path.join(output_folder, "nodes.csv"), index=False)

    mapped_samples = assign_nodes_to_samples(workload_data, nodes, seed=42)
    mapped_samples.to_csv(os.path.join(output_folder, "timeseries.csv"), index=False)

    print("✅ Node mapping completed")
    print(f"📥 Input file used: {input_csv}")
    print(f"🔢 Number of nodes used: {number_of_nodes}")
    print("📁 Saved: nodes.csv")
    print("📁 Saved: timeseries.csv")
    print("\nNode type distribution:")
    print(nodes["node_type"].value_counts())
    print("\nMapped sample distribution:")
    print(mapped_samples["node_type"].value_counts())


if __name__ == "__main__":
    main()