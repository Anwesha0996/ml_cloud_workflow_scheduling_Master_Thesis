# simulation/training_data_generator.py
import os
import numpy as np
import pandas as pd

from workflow_profiles import get_workflow_profile
from sc3_conditions import get_condition


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_inputs(outdir: str):
    nodes_path = os.path.join(outdir, "nodes.csv")
    ts_path = os.path.join(outdir, "timeseries.csv")
    comps_path = os.path.join(outdir, "application_components.csv")
    dag_path = os.path.join(outdir, "application_dag.csv")

    for p in [nodes_path, ts_path, comps_path, dag_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    nodes = pd.read_csv(nodes_path)
    timeseries = pd.read_csv(ts_path)
    comps = pd.read_csv(comps_path)
    dag = pd.read_csv(dag_path)

    nodes["node_id"] = nodes["node_id"].astype(str)
    timeseries["node_id"] = timeseries["node_id"].astype(str)

    return nodes, timeseries, comps, dag


def pick_snapshot_per_node(timeseries, nodes, seed=42):
    rng = np.random.default_rng(seed)
    snaps = []

    for nid in nodes["node_id"].tolist():
        sub = timeseries[timeseries["node_id"] == nid]
        if len(sub) == 0:
            continue
        idx = rng.integers(0, len(sub))
        snaps.append(sub.iloc[[idx]])

    if len(snaps) == 0:
        raise RuntimeError("No snapshots could be picked.")

    snapshots = pd.concat(snaps, ignore_index=True)
    snapshots = snapshots.merge(nodes, on="node_id", how="left")
    return snapshots


def compute_node_cost(row):
    cpu_usage = float(row["cpu"]) / 100.0
    mem_used = float(row["memory"])
    mem_capacity = float(row.get("memory_capacity", 16000.0))
    mem_frac = mem_used / max(mem_capacity, 1.0)

    latency = float(row["latency"])
    bandwidth = float(row["bandwidth"])
    packet_loss = float(row["packet_loss"])
    availability = float(row["availability"])

    load_penalty = 1.0 + 1.8 * cpu_usage + 1.2 * mem_frac
    net_penalty = 1.0 + (latency / 120.0) + (200.0 / max(bandwidth, 1.0)) + (packet_loss / 4.0)
    avail_penalty = 1.0 + (1.0 - availability) * 8.0

    return load_penalty * net_penalty * avail_penalty


def build_node_costs(snapshots, nodes):
    costs = {}

    for _, r in snapshots.iterrows():
        costs[str(r["node_id"])] = compute_node_cost(r)

    avg_cost = float(np.mean(list(costs.values())))

    for nid in nodes["node_id"]:
        if str(nid) not in costs:
            costs[str(nid)] = avg_cost

    return costs


def link_penalty(node_costs, n1, n2, comm_scale=1.0):
    if n1 == n2:
        return 0.0
    return comm_scale * 0.25 * (node_costs[n1] + node_costs[n2])


def component_runtime(node_costs, comp_row, node_id, runtime_scale=1.0, memory_scale=1.0):
    cpu_req = float(comp_row["cpu_request"])
    mem_req = float(comp_row["memory_request"])

    base = 0.7 * cpu_req + (0.00035 * memory_scale) * mem_req
    return runtime_scale * base * node_costs[str(node_id)]


def workflow_metrics_for_assignment(node_costs, comps, dag, assignment,
                                    runtime_scale, comm_scale, memory_scale):

    runtimes = {}
    storage_cost = 0.0

    for _, c in comps.iterrows():
        comp = c["component"]
        node_id = assignment[comp]

        rt = component_runtime(node_costs, c, node_id,
                               runtime_scale, memory_scale)
        runtimes[comp] = rt

        storage_cost += float(c["memory_request"]) * node_costs[str(node_id)] * memory_scale

    compute_cost = float(sum(runtimes.values()))

    network_cost = 0.0
    for _, e in dag.iterrows():
        network_cost += link_penalty(node_costs,
                                     assignment[e["src"]],
                                     assignment[e["dst"]],
                                     comm_scale)

    workflow_time = compute_cost + network_cost

    return {
        "workflow_time": workflow_time,
        "compute_cost": compute_cost,
        "network_cost": float(network_cost),
        "storage_cost": float(storage_cost),
    }


def generate_random_assignments(comps, nodes, n_scenarios, seed=42):
    rng = np.random.default_rng(seed)
    node_list = nodes["node_id"].tolist()
    comp_list = comps["component"].tolist()

    for sid in range(n_scenarios):
        assign = {c: rng.choice(node_list) for c in comp_list}
        yield sid, assign


def main():
    outdir = "simulation"
    ensure_outdir(outdir)

    N_SCENARIOS = int(os.environ.get("N_SCENARIOS", 3000))
    SEED = int(os.environ.get("SEED", 42))

    PROFILE_NAME = os.environ.get("WORKFLOW_PROFILE", "baseline").strip().lower()
    prof = get_workflow_profile(PROFILE_NAME)

    runtime_scale = float(prof["runtime_scale"])
    comm_scale = float(prof["comm_scale"])
    memory_scale = float(prof["memory_scale"])

    COND_NAME = os.environ.get("SC3_CONDITION", "normal").strip().lower()
    cond = get_condition(COND_NAME)

    comm_scale = comm_scale * float(cond["comm_scale"])
    memory_scale = memory_scale * float(cond["memory_scale"])

    print(f"✅ WORKFLOW_PROFILE={PROFILE_NAME} -> {prof}")
    print(f"✅ SC3_CONDITION={COND_NAME} -> {cond}")
    print(f"   Effective: runtime_scale={runtime_scale}, comm_scale={comm_scale}, memory_scale={memory_scale}")
    print(f"   N_SCENARIOS={N_SCENARIOS}, SEED={SEED}")

    nodes, timeseries, comps, dag = load_inputs(outdir)

    snapshots = pick_snapshot_per_node(timeseries, nodes, seed=SEED)
    node_costs = build_node_costs(snapshots, nodes)

    rows = []

    for sid, assign in generate_random_assignments(comps, nodes, N_SCENARIOS, SEED):

        metrics = workflow_metrics_for_assignment(
            node_costs, comps, dag, assign,
            runtime_scale, comm_scale, memory_scale
        )

        metrics["network_fraction"] = metrics["network_cost"] / max(metrics["workflow_time"], 1e-12)

        row = {
            "scenario_id": sid,
            "network_condition": COND_NAME,
            "network_comm_scale": comm_scale,
            "network_memory_scale": memory_scale
        }

        row.update(metrics)

        for c in comps["component"]:
            row[f"place_{c}"] = str(assign[c])

        rows.append(row)

    train = pd.DataFrame(rows)

    out_path = os.path.join(outdir, f"train_nodes_250_{PROFILE_NAME}_{COND_NAME}.csv")
    train.to_csv(out_path, index=False)

    print("✅ Training dataset created!")
    print(f"📁 Saved: {out_path}")

    print("\n--- Head ---")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(train.head(5))

    print("\n--- workflow_time stats ---")
    print(train["workflow_time"].describe())


if __name__ == "__main__":
    main()