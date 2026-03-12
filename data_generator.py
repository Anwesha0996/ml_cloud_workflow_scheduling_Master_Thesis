import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Number of samples to generate
n = 5000

# Simulated system usage
cpu = np.random.randint(5, 95, n)                  # CPU usage %
memory = np.random.randint(100, 16000, n)          # Memory usage in MB
latency = np.random.randint(5, 300, n)             # Network latency ms
bandwidth = np.random.randint(50, 1000, n)         # Bandwidth Mbps
packet_loss = np.random.uniform(0, 5, n)           # Packet loss percentage
availability = np.random.uniform(0.90, 0.999, n)   # Availability score

# Create DataFrame
df = pd.DataFrame({
    "cpu": cpu,
    "memory": memory,
    "latency": latency,
    "bandwidth": bandwidth,
    "packet_loss": packet_loss,
    "availability": availability
})

# Save to CSV
df.to_csv("simulation/data.csv", index=False)

print("✅ Data generated successfully!")
print("📁 File saved as simulation/data.csv")
