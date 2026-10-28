import os
import pandas as pd
import numpy as np

print("🚀 Starting data preparation")

# Path to real dataset
file_path = r"C:\Users\HP\Downloads\machine_usage_days_1_to_8_grouped_300_seconds.csv"

# Check if file exists
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load dataset
df = pd.read_csv(file_path)

print("✅ Dataset loaded successfully")
print("Available columns:", df.columns)
print(df.head())

# Check required columns
required_columns = ["cpu_util_percent", "mem_util_percent"]
missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Create new dataset for simulation
real_df = pd.DataFrame()

# CPU usage (already in percent)
real_df["cpu"] = df["cpu_util_percent"].astype(float)

# Convert memory usage from percent to MB
real_df["memory"] = (df["mem_util_percent"].astype(float) / 100.0) * 16000

# Generate additional network-related features
rng = np.random.default_rng(42)

real_df["latency"] = rng.integers(5, 300, len(df))          # delay in ms
real_df["bandwidth"] = rng.integers(50, 1000, len(df))      # Mbps
real_df["packet_loss"] = rng.uniform(0, 5, len(df))         # percentage
real_df["availability"] = rng.uniform(0.90, 0.999, len(df)) # reliability

# Keep maximum 5000 samples for faster processing
real_df = real_df.sample(n=min(5000, len(real_df)), random_state=42).reset_index(drop=True)

# Save processed dataset
os.makedirs("simulation", exist_ok=True)
output_path = os.path.join("simulation", "data_real.csv")

real_df.to_csv(output_path, index=False)

print("✅ Real dataset prepared for simulation")
print("Saved file:", output_path)
print(real_df.head())
print("Dataset size:", real_df.shape)