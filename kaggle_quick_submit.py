#!/usr/bin/env python3
"""
Quick Kaggle Submission Script for MITSUI Competition
This script provides a fast, memory-efficient deep learning solution
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

print("🚀 MITSUI Deep Learning Quick Submission")
print("=" * 50)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data loading
print("\n📊 Loading data...")
train_data = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train.csv')
test_data = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/test.csv')
train_labels = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train_labels.csv')

print(f"Train data: {train_data.shape}")
print(f"Test data: {test_data.shape}")
print(f"Train labels: {train_labels.shape}")

# Feature preparation
print("\n🔧 Preparing features...")
common_cols = list(set(train_data.columns) & set(test_data.columns))
numeric_cols = [col for col in common_cols if col != 'date_id' and train_data[col].dtype in ['int64', 'float64']]

# Limit features for memory efficiency
if len(numeric_cols) > 300:
    variances = train_data[numeric_cols].var()
    top_features = variances.nlargest(300).index.tolist()
    numeric_cols = top_features
    print(f"Selected top {len(numeric_cols)} features by variance")

# Prepare feature matrices
X_train = train_data[numeric_cols].fillna(0).values
X_test = test_data[numeric_cols].fillna(0).values

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Feature matrices - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")

# Target preparation
print("\n🎯 Preparing targets...")
target_cols = [col for col in train_labels.columns if col != 'date_id']
y_train = train_labels[target_cols].fillna(0).values

# Limit targets for memory efficiency
if len(target_cols) > 50:
    target_cols = target_cols[:50]
    y_train = y_train[:, :50]
    print(f"Limited to first {len(target_cols)} targets for memory efficiency")

print(f"Target matrix: {y_train.shape}")

# Simple LSTM Model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

# Training function
def train_simple_model(X_train, y_train, target_idx, device):
    """Train a simple LSTM model for one target"""
    model = SimpleLSTM(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create sequences
    sequence_length = 5
    X_seq = []
    y_seq = []
    
    for i in range(len(X_train) - sequence_length):
        X_seq.append(X_train[i:i+sequence_length])
        y_seq.append(y_train[i+sequence_length])
    
    X_seq = torch.FloatTensor(np.array(X_seq)).to(device)
    y_seq = torch.FloatTensor(np.array(y_seq)).to(device)
    
    # Train for a few epochs
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_seq)
        loss = criterion(outputs, y_seq.unsqueeze(1))
        loss.backward()
        optimizer.step()
    
    return model

# Generate predictions
print("\n🤖 Training models and generating predictions...")
predictions = {}

for i, target_name in enumerate(target_cols):
    if i % 10 == 0:
        print(f"Processing target {i+1}/{len(target_cols)}: {target_name}")
    
    # Train model for this target
    model = train_simple_model(X_train_scaled, y_train[:, i], i, device)
    
    # Generate predictions
    model.eval()
    with torch.no_grad():
        # Create test sequences
        sequence_length = 5
        X_test_seq = []
        for j in range(len(X_test_scaled) - sequence_length):
            X_test_seq.append(X_test_scaled[j:j+sequence_length])
        
        X_test_seq = torch.FloatTensor(np.array(X_test_seq)).to(device)
        test_preds = model(X_test_seq).cpu().numpy().flatten()
        
        # Pad with zeros for sequence offset
        padded_preds = np.zeros(len(X_test_scaled))
        padded_preds[sequence_length:] = test_preds
        predictions[target_name] = padded_preds
    
    # Clean up
    del model
    gc.collect()

# Create submission file
print("\n💾 Creating submission file...")
submission_data = []

for target_name in predictions.keys():
    preds = predictions[target_name]
    for i, pred in enumerate(preds):
        submission_data.append({
            'date_id': test_data['date_id'].iloc[i],
            'target': target_name,
            'value': pred
        })

submission_df = pd.DataFrame(submission_data)
submission_df.to_csv('submission.csv', index=False)

print(f"✅ Submission file created: submission.csv")
print(f"📊 Submission shape: {submission_df.shape}")
print(f"🎯 Targets included: {len(target_cols)}")
print(f"📈 Features used: {len(numeric_cols)}")

# Display sample
print("\n📋 Sample predictions:")
print(submission_df.head(10))

print("\n🏆 Ready for Kaggle submission!")
print("📁 File: submission.csv")
print("🚀 Upload this file to the competition submission page") 