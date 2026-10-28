# scheduler/ml_scheduler.py
import os
import numpy as np
import pandas as pd
import joblib


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def pick_snapshot_per_node(timeseries: pd.DataFrame, nodes: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Pick one snapshot row per node (represents current node condition).
    """
    rng = np.random.default_rng(seed)
    snaps = []
    for nid in nodes["node_id"]:
        sub = timeseries[timeseries["node_id"] == nid]
        if len(sub) == 0:
            continue
        idx = rng.integers(0, len(sub))
        snaps.append(sub.iloc[[idx]])
    snapshots = pd.concat(snaps, ignore_index=True).merge(nodes, on="node_id", how="left")
    return snapshots


def compute_node_cost(row: pd.Series) -> float:
    """
    Must match training_data_generator.py logic + column names.
    Uses timeseries columns: cpu, memory, latency, bandwidth, packet_loss, availability
    """
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


def build_node_costs(snapshots: pd.DataFrame) -> dict:
    return {r["node_id"]: compute_node_cost(r) for _, r in snapshots.iterrows()}


def assignment_features(dag: pd.DataFrame, node_costs: dict, assignment: dict) -> dict:
    """
    Features must match those used in training (same column names).
    """
    nodes_used = list(assignment.values())
    distinct = len(set(nodes_used))
    costs = [node_costs[n] for n in nodes_used]

    crossings = 0
    coloc = 0
    for _, e in dag.iterrows():
        if assignment[e["src"]] != assignment[e["dst"]]:
            crossings += 1
        else:
            coloc += 1

    return {
        "distinct_nodes_used": distinct,
        "avg_node_cost": float(np.mean(costs)),
        "max_node_cost": float(np.max(costs)),
        "dag_crossings": crossings,
        "colocated_edges": coloc
    }


def random_assignments(components, node_ids, n_candidates: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    for _ in range(n_candidates):
        yield {c: rng.choice(node_ids) for c in components}


def main() -> None:
    # --- Paths ---
    model_path = os.path.join("ml_model", "rf_bundle.joblib")  # ✅ bundle (regressor+classifier)
    nodes_path = os.path.join("simulation", "nodes.csv")
    ts_path = os.path.join("simulation", "timeseries.csv")
    comps_path = os.path.join("simulation", "application_components.csv")
    dag_path = os.path.join("simulation", "application_dag.csv")

    # --- Load model bundle ---
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Missing model: {model_path}. Run ml_model/train_rf.py first.")

    bundle = joblib.load(model_path)

    # Models
    rf_reg = bundle["regressor"]
    rf_cls = bundle["classifier"]

    # Metadata
    feature_cols = bundle["feature_cols"]
    sla_threshold = float(bundle["sla_threshold"])

    # --- Load data ---
    nodes = load_csv(nodes_path)
    timeseries = load_csv(ts_path)
    comps = load_csv(comps_path)
    dag = load_csv(dag_path)

    node_ids = nodes["node_id"].tolist()
    components = comps["component"].tolist()

    # --- Build node costs from one snapshot per node (current state) ---
    snapshots = pick_snapshot_per_node(timeseries, nodes, seed=7)  # separate seed
    node_costs = build_node_costs(snapshots)

    # --- Generate candidate placements + predict for each ---
    N_CANDIDATES = 500
    candidates = []

    for i, assign in enumerate(random_assignments(components, node_ids, N_CANDIDATES, seed=99)):
        feats = assignment_features(dag, node_costs, assign)
        X = pd.DataFrame([feats])[feature_cols]

        pred_time = float(rf_reg.predict(X)[0])
        pred_sla_ok = int(rf_cls.predict(X)[0])

        candidates.append({
            "candidate_id": i,
            "predicted_workflow_time": pred_time,
            "predicted_sla_ok": pred_sla_ok,
            "sla_threshold_used": sla_threshold,
            **assign,
            **feats
        })

    cand_df = pd.DataFrame(candidates)

    # --- Select best placement with SLA constraint ---
    sla_ok_df = cand_df[cand_df["predicted_sla_ok"] == 1].copy()

    if len(sla_ok_df) > 0:
        chosen_df = sla_ok_df.sort_values("predicted_workflow_time", ascending=True)
        best = chosen_df.iloc[0]
        print("\n✅ ML Scheduler selected BEST SLA-OK placement")
    else:
        # Fallback: no candidate meets SLA, choose best time anyway
        chosen_df = cand_df.sort_values("predicted_workflow_time", ascending=True)
        best = chosen_df.iloc[0]
        print("\n⚠️ No SLA-OK placement found among candidates. Falling back to best predicted time.")

    print(f"Predicted workflow time: {best['predicted_workflow_time']:.4f}")
    print(f"Predicted SLA ok (1=yes): {int(best['predicted_sla_ok'])}")
    print(f"SLA threshold used: {best['sla_threshold_used']:.4f}\n")

    for c in components:
        print(f"{c:>18s}  ->  {best[c]}")

    # --- Baseline vs ML summary ---
    baseline_mean = cand_df["predicted_workflow_time"].mean()
    baseline_std = cand_df["predicted_workflow_time"].std()
    best_time = float(best["predicted_workflow_time"])
    improvement = (baseline_mean - best_time) / max(baseline_mean, 1e-9) * 100.0

    print("\n--- Baseline vs ML ---")
    print(f"Random placements (mean ± std): {baseline_mean:.4f} ± {baseline_std:.4f}")
    print(f"ML-selected placement:          {best_time:.4f}")
    print(f"Improvement over random mean:   {improvement:.2f}%")

    # SLA pass-rate among candidates (nice metric)
    sla_pass_rate = cand_df["predicted_sla_ok"].mean() * 100.0
    print(f"SLA pass-rate among candidates: {sla_pass_rate:.2f}%")

    # --- Save results ---
    os.makedirs("scheduler", exist_ok=True)
    out_path = os.path.join("scheduler", "scheduler_results.csv")
    cand_df.sort_values("predicted_workflow_time", ascending=True).to_csv(out_path, index=False)
    print(f"\n📁 Saved candidate results to: {out_path}")


if __name__ == "__main__":
    main()
