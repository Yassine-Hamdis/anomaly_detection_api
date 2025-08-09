# generate_server_data.py
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure data directory exists
Path("data").mkdir(exist_ok=True)

np.random.seed(42)
n_samples = 1000

# Normal ranges
cpu_normal = np.random.normal(50, 10, n_samples)
mem_normal = np.random.normal(60, 15, n_samples)
disk_normal = np.random.normal(100, 30, n_samples)
net_normal = np.random.normal(50, 20, n_samples)

# Anomalies
n_anomalies = 50
cpu_anom = np.random.normal(95, 2, n_anomalies)
mem_anom = np.random.normal(95, 2, n_anomalies)
disk_anom = np.random.normal(500, 50, n_anomalies)
net_anom = np.random.normal(300, 50, n_anomalies)

# Combine
cpu = np.concatenate([cpu_normal, cpu_anom])
mem = np.concatenate([mem_normal, mem_anom])
disk = np.concatenate([disk_normal, disk_anom])
net = np.concatenate([net_normal, net_anom])

df = pd.DataFrame({
    "cpu_util_percent": cpu,
    "memory_util_percent": mem,
    "disk_io": disk,
    "network_io": net
})

df.to_csv("data/server_usage.csv", index=False)
print(f"✅ server_usage.csv generated with {len(df)} rows")
