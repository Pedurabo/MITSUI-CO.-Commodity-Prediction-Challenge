# 🧪 Quick Test Script for Parquet Output
# Run this to verify parquet file creation works

import pandas as pd
import numpy as np

print("🧪 Testing Parquet File Creation...")

# Create sample submission data
sample_data = [
    {'date_id': 1001, 'target': 'target_1', 'value': 0.123},
    {'date_id': 1001, 'target': 'target_2', 'value': 0.456},
    {'date_id': 1002, 'target': 'target_1', 'value': 0.789},
    {'date_id': 1002, 'target': 'target_2', 'value': 0.321},
]

# Create DataFrame
df = pd.DataFrame(sample_data)
print(f"Sample data shape: {df.shape}")
print("Sample data:")
print(df)

# Save as parquet
df.to_parquet('test_submission.parquet', index=False)
print("\n✅ Parquet file created: test_submission.parquet")

# Verify by reading back
df_read = pd.read_parquet('test_submission.parquet')
print(f"✅ Parquet file read back successfully: {df_read.shape}")
print("Read data:")
print(df_read)

print("\n🎯 Parquet output is working correctly!")
print("📁 Your main submission will create: submission.parquet")
