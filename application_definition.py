# simulation/application_definition.py
import os
import pandas as pd


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def define_application_components() -> pd.DataFrame:
    """
    Define a synthetic multi-component application.
    Each component has resource demands and network sensitivity.
    """
    components = [
        {
            "component": "A_preprocessing",
            "cpu_request": 2,            # cores (synthetic)
            "memory_request": 1024,      # MB
            "latency_sensitivity": 0.3,  # low
            "description": "Input data preprocessing"
        },
        {
            "component": "B_inference",
            "cpu_request": 6,
            "memory_request": 4096,
            "latency_sensitivity": 0.9,  # very high
            "description": "ML inference stage"
        },
        {
            "component": "C_postprocessing",
            "cpu_request": 3,
            "memory_request": 2048,
            "latency_sensitivity": 0.6,
            "description": "Result aggregation and formatting"
        },
        {
            "component": "D_storage",
            "cpu_request": 2,
            "memory_request": 1536,
            "latency_sensitivity": 0.2,  # low
            "description": "Database / persistent storage"
        },
        {
            "component": "E_api_gateway",
            "cpu_request": 1,
            "memory_request": 512,
            "latency_sensitivity": 0.7,
            "description": "External API / client interface"
        },
    ]

    return pd.DataFrame(components)


def define_application_dag() -> pd.DataFrame:
    """
    Define the workflow dependencies between components.
    
    DAG structure:
        A_preprocessing -> B_inference -> C_postprocessing
                                          |             |
                                          v             v
                                      D_storage     E_api_gateway
    """
    edges = [
        {"src": "A_preprocessing", "dst": "B_inference"},
        {"src": "B_inference", "dst": "C_postprocessing"},
        {"src": "C_postprocessing", "dst": "D_storage"},
        {"src": "C_postprocessing", "dst": "E_api_gateway"},
    ]

    return pd.DataFrame(edges)


def main() -> None:
    outdir = "simulation"
    ensure_outdir(outdir)

    components = define_application_components()
    dag = define_application_dag()

    comp_path = os.path.join(outdir, "application_components.csv")
    dag_path = os.path.join(outdir, "application_dag.csv")

    components.to_csv(comp_path, index=False)
    dag.to_csv(dag_path, index=False)

    print("✅ Application definition created")
    print(f"📁 Saved: {comp_path}")
    print(f"📁 Saved: {dag_path}")

    print("\n--- Components ---")
    print(components[["component", "cpu_request", "memory_request", "latency_sensitivity"]])

    print("\n--- DAG edges ---")
    print(dag)


if __name__ == "__main__":
    main()
