# simulation/application_definition.py
import os
import pandas as pd


def ensure_output_folder(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_components(workflow_type: str) -> pd.DataFrame:
    workflow_type = workflow_type.strip().lower()

    if workflow_type == "video":
        components = [
            {"component": "take_video", "cpu_request": 2, "memory_request": 1024, "latency_sensitivity": 0.4, "description": "Receive live video stream from camera"},
            {"component": "frame_extraction", "cpu_request": 3, "memory_request": 2048, "latency_sensitivity": 0.5, "description": "Extract frames from video stream"},
            {"component": "preprocessing", "cpu_request": 4, "memory_request": 2048, "latency_sensitivity": 0.6, "description": "Resize, normalize, and clean image frames"},
            {"component": "find_objects", "cpu_request": 8, "memory_request": 4096, "latency_sensitivity": 0.9, "description": "Run object detection model on frames"},
            {"component": "follow_objects", "cpu_request": 4, "memory_request": 2048, "latency_sensitivity": 0.8, "description": "Track detected objects across frames"},
            {"component": "check_what_happens", "cpu_request": 3, "memory_request": 2048, "latency_sensitivity": 0.7, "description": "Analyze events such as congestion or accidents"},
            {"component": "store_data", "cpu_request": 2, "memory_request": 1536, "latency_sensitivity": 0.2, "description": "Store processed results in database"},
            {"component": "notify", "cpu_request": 1, "memory_request": 512, "latency_sensitivity": 0.8, "description": "Send alerts or notifications"},
        ]

    elif workflow_type == "iot":
        components = [
            {"component": "sensor_collector", "cpu_request": 2, "memory_request": 1024, "latency_sensitivity": 0.4, "description": "Collect telemetry from IoT sensors"},
            {"component": "message_broker", "cpu_request": 3, "memory_request": 2048, "latency_sensitivity": 0.6, "description": "Buffer and distribute sensor messages"},
            {"component": "data_cleaning", "cpu_request": 3, "memory_request": 1536, "latency_sensitivity": 0.5, "description": "Remove missing or corrupt sensor values"},
            {"component": "feature_extraction", "cpu_request": 4, "memory_request": 2048, "latency_sensitivity": 0.6, "description": "Generate features from sensor data"},
            {"component": "prediction_model", "cpu_request": 6, "memory_request": 3072, "latency_sensitivity": 0.8, "description": "Run anomaly or prediction model"},
            {"component": "rule_engine", "cpu_request": 3, "memory_request": 1536, "latency_sensitivity": 0.7, "description": "Apply decision rules on model output"},
            {"component": "database_cloud_storage", "cpu_request": 2, "memory_request": 2048, "latency_sensitivity": 0.3, "description": "Store processed data in cloud database"},
            {"component": "dashboard", "cpu_request": 1, "memory_request": 1024, "latency_sensitivity": 0.7, "description": "Visualize sensor and analytics results"},
            {"component": "alert", "cpu_request": 1, "memory_request": 512, "latency_sensitivity": 0.8, "description": "Generate alert for abnormal conditions"},
        ]

    elif workflow_type == "cicd":
        components = [
            {"component": "source_fetch", "cpu_request": 2, "memory_request": 1024, "latency_sensitivity": 0.3, "description": "Fetch source code from repository"},
            {"component": "build", "cpu_request": 5, "memory_request": 3072, "latency_sensitivity": 0.4, "description": "Compile and build application"},
            {"component": "unit_test", "cpu_request": 4, "memory_request": 2048, "latency_sensitivity": 0.4, "description": "Execute unit tests"},
            {"component": "integration_test", "cpu_request": 5, "memory_request": 3072, "latency_sensitivity": 0.5, "description": "Execute integration tests"},
            {"component": "static_analysis", "cpu_request": 3, "memory_request": 2048, "latency_sensitivity": 0.3, "description": "Run code quality and security checks"},
            {"component": "image_build", "cpu_request": 4, "memory_request": 3072, "latency_sensitivity": 0.5, "description": "Build container image"},
            {"component": "registry_push", "cpu_request": 2, "memory_request": 1024, "latency_sensitivity": 0.7, "description": "Push image to container registry"},
            {"component": "deployment", "cpu_request": 3, "memory_request": 2048, "latency_sensitivity": 0.8, "description": "Deploy application to Kubernetes cluster"},
            {"component": "verification", "cpu_request": 2, "memory_request": 1024, "latency_sensitivity": 0.7, "description": "Verify deployment health and status"},
        ]

    else:
        raise ValueError("Unknown workflow type. Choose one of: ['video', 'iot', 'cicd']")

    return pd.DataFrame(components)


def get_workflow_edges(workflow_type: str) -> pd.DataFrame:
    workflow_type = workflow_type.strip().lower()

    if workflow_type == "video":
        edges = [
            {"src": "take_video", "dst": "frame_extraction"},
            {"src": "frame_extraction", "dst": "preprocessing"},
            {"src": "preprocessing", "dst": "find_objects"},
            {"src": "find_objects", "dst": "follow_objects"},
            {"src": "follow_objects", "dst": "check_what_happens"},
            {"src": "check_what_happens", "dst": "store_data"},
            {"src": "check_what_happens", "dst": "notify"},
        ]

    elif workflow_type == "iot":
        edges = [
            {"src": "sensor_collector", "dst": "message_broker"},
            {"src": "message_broker", "dst": "data_cleaning"},
            {"src": "data_cleaning", "dst": "feature_extraction"},
            {"src": "feature_extraction", "dst": "prediction_model"},
            {"src": "prediction_model", "dst": "rule_engine"},
            {"src": "rule_engine", "dst": "database_cloud_storage"},
            {"src": "database_cloud_storage", "dst": "dashboard"},
            {"src": "database_cloud_storage", "dst": "alert"},
        ]

    elif workflow_type == "cicd":
        edges = [
            {"src": "source_fetch", "dst": "build"},
            {"src": "build", "dst": "unit_test"},
            {"src": "unit_test", "dst": "integration_test"},
            {"src": "integration_test", "dst": "static_analysis"},
            {"src": "static_analysis", "dst": "image_build"},
            {"src": "image_build", "dst": "registry_push"},
            {"src": "registry_push", "dst": "deployment"},
            {"src": "deployment", "dst": "verification"},
        ]

    else:
        raise ValueError("Unknown workflow type. Choose one of: ['video', 'iot', 'cicd']")

    return pd.DataFrame(edges)


def save_one_workflow(output_folder: str, workflow_type: str) -> None:
    components = get_components(workflow_type)
    dag = get_workflow_edges(workflow_type)

    components_file = os.path.join(output_folder, f"application_components_{workflow_type}.csv")
    dag_file = os.path.join(output_folder, f"application_dag_{workflow_type}.csv")

    components.to_csv(components_file, index=False)
    dag.to_csv(dag_file, index=False)

    print(f"\n✅ Saved workflow: {workflow_type}")
    print(f"📁 {components_file}")
    print(f"📁 {dag_file}")
    print("\n--- Components ---")
    print(components[["component", "cpu_request", "memory_request", "latency_sensitivity"]])
    print("\n--- DAG ---")
    print(dag)


def main() -> None:
    output_folder = "simulation"
    ensure_output_folder(output_folder)

    workflow_types = ["video", "iot", "cicd"]

    for workflow_type in workflow_types:
        save_one_workflow(output_folder, workflow_type)

    print("\n✅ All 3 workflow files created successfully.")


if __name__ == "__main__":
    main()