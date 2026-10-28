import os
import numpy as np
import pandas as pd

from workflow_profiles import get_workflow_profile
from sc3_conditions import get_condition


def ensure_output_folder(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_input_files(output_folder: str):
    nodes_file = os.path.join(output_folder, "nodes.csv")
    time_series_file = os.path.join(output_folder, "timeseries.csv")
    components_file = os.path.join(output_folder, "application_components.csv")
    workflow_file = os.path.join(output_folder, "application_dag.csv")

    for file_path in [nodes_file, time_series_file, components_file, workflow_file]:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Missing required file: {file_path}")

    nodes = pd.read_csv(nodes_file)
    time_series = pd.read_csv(time_series_file)
    components = pd.read_csv(components_file)
    workflow = pd.read_csv(workflow_file)

    nodes["node_id"] = nodes["node_id"].astype(str)
    time_series["node_id"] = time_series["node_id"].astype(str)

    return nodes, time_series, components, workflow


def select_one_snapshot_per_node(
    time_series: pd.DataFrame,
    nodes: pd.DataFrame,
    seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    selected_rows = []

    for node_id in nodes["node_id"].tolist():
        matching_rows = time_series[time_series["node_id"] == node_id]
        if len(matching_rows) == 0:
            continue

        random_index = rng.integers(0, len(matching_rows))
        selected_rows.append(matching_rows.iloc[[random_index]])

    if len(selected_rows) == 0:
        raise RuntimeError("No node snapshots could be selected.")

    snapshots = pd.concat(selected_rows, ignore_index=True)
    snapshots = snapshots.merge(nodes, on="node_id", how="left")
    return snapshots


def calculate_node_condition_score(row: pd.Series) -> float:
    """
    This tells how 'busy' or 'difficult' a node currently is.
    Higher score = more pressure on that node.
    """
    cpu_usage = float(row["cpu"]) / 100.0
    memory_used = float(row["memory"])
    memory_capacity = float(row.get("memory_capacity", 16000.0))
    memory_fraction = memory_used / max(memory_capacity, 1.0)

    latency = float(row["latency"])
    bandwidth = float(row["bandwidth"])
    packet_loss = float(row["packet_loss"])
    availability = float(row["availability"])

    resource_pressure = 1.0 + 1.8 * cpu_usage + 1.2 * memory_fraction
    network_pressure = 1.0 + (latency / 120.0) + (200.0 / max(bandwidth, 1.0)) + (packet_loss / 4.0)
    reliability_pressure = 1.0 + (1.0 - availability) * 8.0

    return resource_pressure * network_pressure * reliability_pressure


def build_node_condition_scores(snapshots: pd.DataFrame, nodes: pd.DataFrame) -> dict:
    node_condition_scores = {}

    for _, row in snapshots.iterrows():
        node_condition_scores[str(row["node_id"])] = calculate_node_condition_score(row)

    average_score = float(np.mean(list(node_condition_scores.values())))

    for node_id in nodes["node_id"]:
        if str(node_id) not in node_condition_scores:
            node_condition_scores[str(node_id)] = average_score

    return node_condition_scores


def calculate_communication_delay(
    node_condition_scores: dict,
    source_node: str,
    target_node: str,
    communication_scale: float = 1.0
) -> float:
    if source_node == target_node:
        return 0.0

    return communication_scale * 0.25 * (
        node_condition_scores[source_node] + node_condition_scores[target_node]
    )


def calculate_component_processing_time(
    node_condition_scores: dict,
    component_row: pd.Series,
    node_id: str,
    runtime_scale: float = 1.0,
    memory_scale: float = 1.0
) -> float:
    cpu_request = float(component_row["cpu_request"])
    memory_request = float(component_row["memory_request"])

    base_processing_time = 0.7 * cpu_request + (0.00035 * memory_scale) * memory_request
    return runtime_scale * base_processing_time * node_condition_scores[str(node_id)]


def calculate_workflow_metrics(
    node_condition_scores: dict,
    components: pd.DataFrame,
    workflow: pd.DataFrame,
    assignment: dict,
    runtime_scale: float,
    communication_scale: float,
    memory_scale: float
) -> dict:
    component_processing_times = {}
    total_memory_use = 0.0

    for _, component_row in components.iterrows():
        component_name = component_row["component"]
        node_id = assignment[component_name]

        processing_time = calculate_component_processing_time(
            node_condition_scores,
            component_row,
            node_id,
            runtime_scale,
            memory_scale
        )

        component_processing_times[component_name] = processing_time

        total_memory_use += (
            float(component_row["memory_request"])
            * node_condition_scores[str(node_id)]
            * memory_scale
        )

    total_processing_time = float(sum(component_processing_times.values()))

    total_communication_delay = 0.0
    for _, edge in workflow.iterrows():
        total_communication_delay += calculate_communication_delay(
            node_condition_scores,
            assignment[edge["src"]],
            assignment[edge["dst"]],
            communication_scale
        )

    total_workflow_time = total_processing_time + total_communication_delay

    return {
        "workflow_time": total_workflow_time,
        "processing_time": total_processing_time,
        "communication_delay": float(total_communication_delay),
        "memory_use": float(total_memory_use),
    }


def generate_random_component_assignments(
    components: pd.DataFrame,
    nodes: pd.DataFrame,
    number_of_scenarios: int,
    seed: int = 42
):
    rng = np.random.default_rng(seed)
    node_ids = nodes["node_id"].tolist()
    component_names = components["component"].tolist()

    for scenario_id in range(number_of_scenarios):
        assignment = {component: rng.choice(node_ids) for component in component_names}
        yield scenario_id, assignment


def main():
    output_folder = "simulation"
    ensure_output_folder(output_folder)

    number_of_scenarios = int(os.environ.get("N_SCENARIOS", 3000))
    seed = int(os.environ.get("SEED", 42))

    workflow_profile_name = os.environ.get("WORKFLOW_PROFILE", "baseline").strip().lower()
    workflow_profile = get_workflow_profile(workflow_profile_name)

    runtime_scale = float(workflow_profile["runtime_scale"])
    communication_scale = float(workflow_profile["comm_scale"])
    memory_scale = float(workflow_profile["memory_scale"])

    condition_name = os.environ.get("SC3_CONDITION", "normal").strip().lower()
    condition = get_condition(condition_name)

    communication_scale = communication_scale * float(condition["comm_scale"])
    memory_scale = memory_scale * float(condition["memory_scale"])

    print(f"✅ Workflow profile: {workflow_profile_name} -> {workflow_profile}")
    print(f"✅ Environment condition: {condition_name} -> {condition}")
    print(
        f"   Final settings: runtime_scale={runtime_scale}, "
        f"communication_scale={communication_scale}, memory_scale={memory_scale}"
    )
    print(f"   Number of scenarios={number_of_scenarios}, seed={seed}")

    nodes, time_series, components, workflow = load_input_files(output_folder)

    # This is the important new part:
    # count how many nodes are currently in nodes.csv
    number_of_nodes = len(nodes)

    snapshots = select_one_snapshot_per_node(time_series, nodes, seed=seed)
    node_condition_scores = build_node_condition_scores(snapshots, nodes)

    rows = []

    for scenario_id, assignment in generate_random_component_assignments(
        components,
        nodes,
        number_of_scenarios,
        seed
    ):
        metrics = calculate_workflow_metrics(
            node_condition_scores,
            components,
            workflow,
            assignment,
            runtime_scale,
            communication_scale,
            memory_scale
        )

        metrics["communication_fraction"] = (
            metrics["communication_delay"] / max(metrics["workflow_time"], 1e-12)
        )

        row = {
            "scenario_id": scenario_id,
            "environment_condition": condition_name,
            "communication_scale": communication_scale,
            "memory_scale": memory_scale
        }

        row.update(metrics)

        for component_name in components["component"]:
            row[f"assigned_node_{component_name}"] = str(assignment[component_name])

        rows.append(row)

    training_data = pd.DataFrame(rows)

    # Save file with the real node count
    output_file = os.path.join(
        output_folder,
        f"train_nodes_{number_of_nodes}_{workflow_profile_name}_{condition_name}_seed{seed}.csv"
    )
    training_data.to_csv(output_file, index=False)

    print("✅ Training dataset created")
    print(f"📁 Saved: {output_file}")

    print("\n--- First rows ---")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(training_data.head(5))

    print("\n--- Workflow time summary ---")
    print(training_data["workflow_time"].describe())


if __name__ == "__main__":
    main()