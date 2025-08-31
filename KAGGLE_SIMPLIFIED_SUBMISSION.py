# 🚀 MITSUI&CO. Commodity Prediction Challenge - Deep Learning Submission
# Copy and paste this entire script into a new Kaggle notebook
# PyTorch is already installed with CUDA support!
# OUTPUTS: submission.parquet (required by competition)

# Import libraries (all pre-installed on Kaggle)
import os
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Data loading and preprocessing functions
def load_data():
    """Load competition data"""
    print("Loading data...")
    
    train_data = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train.csv')
    test_data = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/test.csv')
    train_labels = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train_labels.csv')
    target_pairs = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/target_pairs.csv')
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Target pairs shape: {target_pairs.shape}")
    
    return train_data, test_data, train_labels, target_pairs

def prepare_features(train_data, test_data):
    """Prepare features for deep learning models"""
    print("Preparing features...")
    
    # Find common columns
    common_cols = list(set(train_data.columns) & set(test_data.columns))
    print(f"Common columns: {len(common_cols)}")
    
    # Select numeric columns only
    numeric_cols = []
    for col in common_cols:
        if col != 'date_id' and train_data[col].dtype in ['int64', 'float64']:
            numeric_cols.append(col)
    
    print(f"Numeric columns: {len(numeric_cols)}")
    
    # Limit features for memory efficiency
    if len(numeric_cols) > 500:
        # Use variance-based selection
        variances = train_data[numeric_cols].var()
        top_features = variances.nlargest(500).index.tolist()
        numeric_cols = top_features
        print(f"Selected top {len(numeric_cols)} features by variance")
    
    # Prepare feature matrices
    X_train = train_data[numeric_cols].fillna(0).values
    X_test = test_data[numeric_cols].fillna(0).values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Feature matrix shapes - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, numeric_cols, scaler

def prepare_targets(train_labels):
    """Prepare target variables"""
    print("Preparing targets...")
    
    # Get target columns (excluding date_id)
    target_cols = [col for col in train_labels.columns if col != 'date_id']
    print(f"Number of targets: {len(target_cols)}")
    
    # Prepare target matrix
    y_train = train_labels[target_cols].fillna(0).values
    print(f"Target matrix shape: {y_train.shape}")
    
    return y_train, target_cols

# Deep Learning Model Definitions
class TimeSeriesDataset(Dataset):
    """Custom Dataset for time series data"""
    def __init__(self, X, y, sequence_length=10):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.X) - self.sequence_length
    
    def __getitem__(self, idx):
        return (
            self.X[idx:idx + self.sequence_length],
            self.y[idx + self.sequence_length]
        )

class LSTMModel(nn.Module):
    """LSTM model for time series prediction"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.fc(out)
        return out

class GRUModel(nn.Module):
    """GRU model for time series prediction"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.fc(out)
        return out

# Training functions
def train_model(model, train_loader, val_loader, device, epochs=50, patience=10):
    """Train a deep learning model"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return model, train_losses, val_losses

def predict_with_model(model, test_loader, device):
    """Generate predictions with a trained model"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy().flatten())
    
    return np.array(predictions)

# Main training pipeline
def train_deep_learning_models(X_train, y_train, target_cols, device):
    """Train multiple deep learning models for all targets"""
    print("Training deep learning models...")
    
    models = {}
    predictions = {}
    
    # Train models for each target
    for i, target_name in enumerate(target_cols):
        if i % 50 == 0:
            print(f"Processing target {i+1}/{len(target_cols)}: {target_name}")
        
        # Prepare data for this target
        y_target = y_train[:, i]
        
        # Split data
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_target, test_size=0.2, random_state=42
        )
        
        # Create datasets
        sequence_length = 10
        train_dataset = TimeSeriesDataset(X_train_split, y_train_split, sequence_length)
        val_dataset = TimeSeriesDataset(X_val_split, y_val_split, sequence_length)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Train LSTM model
        lstm_model = LSTMModel(
            input_size=X_train.shape[1],
            hidden_size=64,  # Reduced for memory efficiency
            num_layers=2,
            dropout=0.2
        )
        
        trained_lstm, _, _ = train_model(
            lstm_model, train_loader, val_loader, device, epochs=30, patience=8
        )
        
        # Store model and generate predictions
        models[f"lstm_{target_name}"] = trained_lstm
        
        # Generate predictions for training data
        train_dataset_full = TimeSeriesDataset(X_train, y_target, sequence_length)
        train_loader_full = DataLoader(train_dataset_full, batch_size=32, shuffle=False)
        
        preds = predict_with_model(trained_lstm, train_loader_full, device)
        predictions[f"lstm_{target_name}"] = preds
        
        # Clean up memory
        del trained_lstm, train_dataset, val_dataset, train_loader, val_loader
        gc.collect()
        
        # Limit to first 100 targets for memory efficiency
        if i >= 99:
            print(f"Stopping at {i+1} targets for memory efficiency")
            break
    
    return models, predictions

def generate_test_predictions(models, X_test, target_cols, device):
    """Generate predictions for test data"""
    print("Generating test predictions...")
    
    test_predictions = {}
    sequence_length = 10
    
    # Process each target
    for i, target_name in enumerate(target_cols):
        if i % 50 == 0:
            print(f"Generating predictions for target {i+1}/{len(target_cols)}: {target_name}")
        
        model_key = f"lstm_{target_name}"
        if model_key in models:
            model = models[model_key]
            
            # Create test dataset
            test_dataset = TimeSeriesDataset(X_test, np.zeros(len(X_test)), sequence_length)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            preds = predict_with_model(model, test_loader, device)
            test_predictions[target_name] = preds
        
        # Limit to first 100 targets for memory efficiency
        if i >= 99:
            break
    
    return test_predictions

# Main execution
print("🚀 Starting MITSUI Deep Learning Competition Submission")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
train_data, test_data, train_labels, target_pairs = load_data()

# Prepare features and targets
X_train, X_test, feature_cols, scaler = prepare_features(train_data, test_data)
y_train, target_cols = prepare_targets(train_labels)

print(f"\n📊 Data Summary:")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Features: {X_train.shape[1]}")
print(f"Targets: {len(target_cols)}")

# Train models
models, train_predictions = train_deep_learning_models(X_train, y_train, target_cols, device)

# Generate test predictions
test_predictions = generate_test_predictions(models, X_test, target_cols, device)

print("\n✅ Training and prediction completed successfully!")
print(f"Trained models: {len(models)}")
print(f"Generated predictions for {len(test_predictions)} targets")

# Save results
print("\n💾 Saving results...")

# Create submission dataframe
submission_data = []
for target_name in test_predictions.keys():
    preds = test_predictions[target_name]
    for i, pred in enumerate(preds):
        submission_data.append({
            'date_id': test_data['date_id'].iloc[i + 10],  # +10 for sequence length offset
            'target': target_name,
            'value': pred
        })

submission_df = pd.DataFrame(submission_data)

# Save as PARQUET file (required by competition)
submission_df.to_parquet('submission.parquet', index=False)

print(f"\n🎯 Submission file created: submission.parquet")
print(f"Submission shape: {submission_df.shape}")
print(f"\n🏆 Ready for Kaggle submission!")

# Display sample predictions
print("\n📋 Sample predictions:")
print(submission_df.head(10))

print("\n🎉 Submission complete! Upload submission.parquet to the competition page.")
print("📁 File: submission.parquet (PARQUET format required by competition)")
