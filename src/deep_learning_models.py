import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# PyTorch Models

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

class TransformerModel(nn.Module):
    """Transformer model for time series prediction"""
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.dropout(x)
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
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

class HybridModel(nn.Module):
    """Hybrid model combining CNN, LSTM, and attention"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(HybridModel, self).__init__()
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # CNN feature extraction
        x = x.transpose(1, 2)  # (batch, features, sequence)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)  # (batch, sequence, features)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        x = attn_out.mean(dim=1)
        
        # Output layers
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# TensorFlow/Keras Models

def create_lstm_keras(input_shape, units=128, layers=2, dropout=0.2):
    """Create LSTM model using Keras"""
    model = keras.Sequential()
    
    for i in range(layers):
        if i == 0:
            model.add(keras.layers.LSTM(units, return_sequences=True if i < layers-1 else False, 
                                       input_shape=input_shape))
        else:
            model.add(keras.layers.LSTM(units, return_sequences=True if i < layers-1 else False))
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1))
    
    return model

def create_gru_keras(input_shape, units=128, layers=2, dropout=0.2):
    """Create GRU model using Keras"""
    model = keras.Sequential()
    
    for i in range(layers):
        if i == 0:
            model.add(keras.layers.GRU(units, return_sequences=True if i < layers-1 else False, 
                                      input_shape=input_shape))
        else:
            model.add(keras.layers.GRU(units, return_sequences=True if i < layers-1 else False))
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1))
    
    return model

def create_transformer_keras(input_shape, d_model=128, num_heads=8, num_layers=4, dropout=0.1):
    """Create Transformer model using Keras"""
    inputs = keras.layers.Input(shape=input_shape)
    
    # Input projection
    x = keras.layers.Dense(d_model)(inputs)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Positional encoding - ensure it matches the projected dimension
    pos_encoding = positional_encoding_1d(inputs.shape[1], d_model)
    # Reshape pos_encoding to match x shape for broadcasting
    pos_encoding = tf.reshape(pos_encoding, (1, inputs.shape[1], d_model))
    x = x + pos_encoding
    
    # Transformer blocks
    for _ in range(num_layers):
        # Multi-head attention
        attention_output = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout
        )(x, x)
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
        
        # Feed-forward network
        ffn = keras.Sequential([
            keras.layers.Dense(d_model * 4, activation='relu'),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(d_model)
        ])
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn(x))
    
    # Global average pooling and output
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(1)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def positional_encoding_1d(length, depth):
    """Create 1D positional encoding"""
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth
    
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates
    
    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)
    
    # Ensure the encoding has the right shape for broadcasting
    pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)
    pos_encoding = tf.expand_dims(pos_encoding, axis=0)  # Add batch dimension
    
    return pos_encoding

# Model Training Classes

class PyTorchTrainer:
    """PyTorch model trainer"""
    def __init__(self, model, device=device):
        self.model = model.to(device)
        self.device = device
        self.scaler = StandardScaler()
        
    def prepare_data(self, X, y, sequence_length=10, batch_size=32):
        """Prepare data for training"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create dataset and dataloader
        dataset = TimeSeriesDataset(X_scaled, y, sequence_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader, self.scaler
        
    def train(self, train_loader, val_loader=None, epochs=100, lr=0.001, patience=20):
        """Train the model"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//2, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
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
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        loss = criterion(outputs.squeeze(), batch_y)
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
        """Make predictions"""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        
        predictions = []
        with torch.no_grad():
            for i in range(sequence_length, len(X_scaled)):
                x = torch.FloatTensor(X_scaled[i-sequence_length:i]).unsqueeze(0).to(self.device)
                pred = self.model(x)
                predictions.append(pred.cpu().numpy()[0, 0])
        
        return np.array(predictions)

class KerasTrainer:
    """Keras model trainer"""
    def __init__(self, model):
        self.model = model
        self.scaler = StandardScaler()
        
    def prepare_data(self, X, y, sequence_length=10):
        """Prepare data for training"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = [], []
        for i in range(sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i-sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq), self.scaler
        
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, patience=20):
        """Train the model"""
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = []
        if X_val is not None:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True
                )
            )
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=patience//2,
                    min_lr=1e-6
                )
            )
        
        # Training
        if X_val is not None:
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
        
        return history
    
    def predict(self, X, sequence_length=10):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        
        X_seq = []
        for i in range(sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i-sequence_length:i])
        
        X_seq = np.array(X_seq)
        predictions = self.model.predict(X_seq, verbose=0)
        
        return predictions.flatten()

# Model Factory Functions

def get_deep_learning_model(model_type, input_size, **kwargs):
    """Get deep learning model by type"""
    if model_type == 'lstm_pytorch':
        return LSTMModel(input_size, **kwargs)
    elif model_type == 'gru_pytorch':
        return GRUModel(input_size, **kwargs)
    elif model_type == 'transformer_pytorch':
        return TransformerModel(input_size, **kwargs)
    elif model_type == 'hybrid_pytorch':
        return HybridModel(input_size, **kwargs)
    elif model_type == 'lstm_keras':
        return create_lstm_keras((None, input_size), **kwargs)
    elif model_type == 'gru_keras':
        return create_gru_keras((None, input_size), **kwargs)
    elif model_type == 'transformer_keras':
        return create_transformer_keras((None, input_size), **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_trainer(model, framework='pytorch'):
    """Get appropriate trainer for the model"""
    if framework == 'pytorch':
        return PyTorchTrainer(model)
    elif framework == 'keras':
        return KerasTrainer(model)
    else:
        raise ValueError(f"Unknown framework: {framework}")
