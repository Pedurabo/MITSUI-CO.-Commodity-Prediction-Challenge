"""
Advanced Deep Learning Techniques Module
Implements multi-task learning, attention mechanisms, and advanced architectures
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class MultiTaskDataset(Dataset):
    """Dataset for multi-task learning with multiple targets"""
    
    def __init__(self, X, y_dict, sequence_length=10):
        self.X = torch.FloatTensor(X)
        self.y_dict = {k: torch.FloatTensor(v) for k, v in y_dict.items()}
        self.sequence_length = sequence_length
        self.target_names = list(y_dict.keys())
        
    def __len__(self):
        return len(self.X) - self.sequence_length
        
    def __getitem__(self, idx):
        x = self.X[idx:idx + self.sequence_length]
        y = {name: self.y_dict[name][idx + self.sequence_length] for name in self.target_names}
        return x, y

class MultiTaskLSTM(nn.Module):
    """Multi-task LSTM model for predicting multiple targets simultaneously"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, num_targets=5):
        super(MultiTaskLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_targets = num_targets
        
        # Shared LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Task-specific output layers
        self.task_heads = nn.ModuleDict()
        for i in range(num_targets):
            self.task_heads[f'target_{i}'] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            )
        
        # Shared attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = attn_out.mean(dim=1)
        
        # Task-specific predictions
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(pooled)
        
        return outputs

class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer for time series forecasting"""
    
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super(TemporalFusionTransformer, self).__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Temporal attention layers
        self.temporal_layers = nn.ModuleList([
            TemporalAttentionLayer(d_model, nhead, dropout) 
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, 1)
        
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Temporal attention processing
        for layer in self.temporal_layers:
            x = layer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class TemporalAttentionLayer(nn.Module):
    """Temporal attention layer with residual connections"""
    
    def __init__(self, d_model, nhead, dropout=0.1):
        super(TemporalAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class AttentionLSTM(nn.Module):
    """LSTM with attention mechanism for better sequence understanding"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            return_sequences=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Attention weights
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Output layer
        out = self.dropout(attended_output)
        out = self.fc(out)
        
        return out

class MultiTaskTrainer:
    """Trainer for multi-task learning models"""
    
    def __init__(self, model, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = model.to(self.device)
        self.scaler = None
        
    def prepare_data(self, X, y_dict, sequence_length=10, batch_size=32):
        """Prepare data for multi-task training"""
        from sklearn.preprocessing import StandardScaler
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Create dataset and dataloader
        dataset = MultiTaskDataset(X_scaled, y_dict, sequence_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader, self.scaler
        
    def train(self, train_loader, val_loader=None, epochs=100, lr=0.001, patience=20):
        """Train the multi-task model"""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//2, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y_dict in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y_dict = {k: v.to(self.device) for k, v in batch_y_dict.items()}
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                # Calculate loss for each task
                loss = 0
                for task_name in outputs.keys():
                    if task_name in batch_y_dict:
                        task_loss = criterion(outputs[task_name].squeeze(), batch_y_dict[task_name])
                        loss += task_loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y_dict in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y_dict = {k: v.to(self.device) for k, v in batch_y_dict.items()}
                        
                        outputs = self.model(batch_X)
                        
                        # Calculate validation loss
                        loss = 0
                        for task_name in outputs.keys():
                            if task_name in batch_y_dict:
                                task_loss = criterion(outputs[task_name].squeeze(), batch_y_dict[task_name])
                                loss += task_loss
                        
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}")
        
        return train_losses, val_losses if val_loader else None
    
    def predict(self, X, sequence_length=10):
        """Make predictions with the multi-task model"""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        with torch.no_grad():
            for i in range(sequence_length, len(X_scaled)):
                x = torch.FloatTensor(X_scaled[i-sequence_length:i]).unsqueeze(0).to(self.device)
                pred = self.model(x)
                
                # Store predictions for each task
                for task_name, task_pred in pred.items():
                    if task_name not in predictions:
                        predictions[task_name] = []
                    predictions[task_name].append(task_pred.cpu().numpy()[0, 0])
        
        # Convert to numpy arrays
        for task_name in predictions:
            predictions[task_name] = np.array(predictions[task_name])
        
        return predictions

def create_multi_task_model(input_size, num_targets, model_type='lstm', **kwargs):
    """Factory function to create multi-task models"""
    if model_type == 'lstm':
        return MultiTaskLSTM(input_size, num_targets=num_targets, **kwargs)
    elif model_type == 'attention_lstm':
        return AttentionLSTM(input_size, **kwargs)
    elif model_type == 'temporal_fusion':
        return TemporalFusionTransformer(input_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_advanced_ensemble(models_dict, ensemble_method='weighted_average'):
    """Create ensemble from multiple advanced models"""
    
    class AdvancedEnsemble(nn.Module):
        def __init__(self, models, method='weighted_average'):
            super(AdvancedEnsemble, self).__init__()
            self.models = nn.ModuleList(models)
            self.method = method
            
            if method == 'weighted_average':
                # Learnable weights for each model
                self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
            
        def forward(self, x):
            outputs = []
            for model in self.models:
                outputs.append(model(x))
            
            if self.method == 'simple_average':
                # Simple averaging
                if isinstance(outputs[0], dict):
                    # Multi-task case
                    ensemble_output = {}
                    for key in outputs[0].keys():
                        ensemble_output[key] = torch.stack([out[key] for out in outputs]).mean(0)
                    return ensemble_output
                else:
                    # Single task case
                    return torch.stack(outputs).mean(0)
            
            elif self.method == 'weighted_average':
                # Weighted averaging
                weights = F.softmax(self.weights, dim=0)
                if isinstance(outputs[0], dict):
                    # Multi-task case
                    ensemble_output = {}
                    for key in outputs[0].keys():
                        weighted_sum = sum(w * out[key] for w, out in zip(weights, outputs))
                        ensemble_output[key] = weighted_sum
                    return ensemble_output
                else:
                    # Single task case
                    weighted_sum = sum(w * out for w, out in zip(weights, outputs))
                    return weighted_sum
    
    return AdvancedEnsemble(list(models_dict.values()), ensemble_method)
