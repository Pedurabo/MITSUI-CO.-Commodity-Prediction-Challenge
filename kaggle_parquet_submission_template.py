# MITSUI&CO. Commodity Prediction Challenge - Parquet Submission Template
# This template generates submission.parquet file as required by the competition

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Essential ML Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

print("🚀 MITSUI Commodity Prediction Challenge - Parquet Submission")
print(f"⏰ Started at: {datetime.now()}")

# ============================================================================
# SETUP AND DATA LOADING
# ============================================================================

# Setup paths
competition_path = '/kaggle/input/mitsui-commodity-prediction-challenge'

try:
    # Load data
    print("📊 Loading competition data...")
    target_pairs = pd.read_csv(f'{competition_path}/target_pairs.csv')
    train_data = pd.read_csv(f'{competition_path}/train.csv')
    train_labels = pd.read_csv(f'{competition_path}/train_labels.csv')
    test_data = pd.read_csv(f'{competition_path}/test.csv')
    
    print(f"📈 Training data: {train_data.shape[0]} samples, {train_data.shape[1]} features")
    print(f"📊 Test data: {test_data.shape[0]} samples, {test_data.shape[1]} features")
    print(f"🎯 Targets: {len(target_pairs)} target pairs")
    
except Exception as e:
    print(f"❌ Error loading data: {str(e)}")
    raise

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_features(df):
    """Create essential features for prediction."""
    features = df.copy()
    
    # Get key columns
    price_cols = [col for col in df.columns if 'Close' in col or 'close' in col]
    fx_cols = [col for col in df.columns if col.startswith('FX_')]
    
    print(f"🔧 Creating features for {len(price_cols)} price columns and {len(fx_cols)} FX pairs")
    
    # Price-based features
    for col in price_cols + fx_cols:
        if col in df.columns:
            # Returns and ratios
            features[f'{col}_pct_change'] = df[col].pct_change()
            features[f'{col}_ma_5'] = df[col].rolling(window=5).mean()
            features[f'{col}_ma_20'] = df[col].rolling(window=20).mean()
            features[f'{col}_ma_5_ratio'] = df[col] / features[f'{col}_ma_5']
            features[f'{col}_ma_20_ratio'] = df[col] / features[f'{col}_ma_20']
    
    # Cross-asset features
    if 'LME_AH_Close' in df.columns and 'LME_ZS_Close' in df.columns:
        features['LME_AH_ZS_spread'] = df['LME_AH_Close'] - df['LME_ZS_Close']
    
    if 'JPX_Gold_Standard_Futures_Close' in df.columns and 'JPX_Platinum_Standard_Futures_Close' in df.columns:
        features['Gold_Platinum_spread'] = df['JPX_Gold_Standard_Futures_Close'] - df['JPX_Platinum_Standard_Futures_Close']
    
    # Clean data
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return features

# ============================================================================
# MODEL TRAINING
# ============================================================================

class TargetPredictor:
    """Manages prediction models for each target."""
    
    def __init__(self, target_pairs):
        self.target_pairs = target_pairs
        self.models = {}
        self.features = {}
        
    def get_target_features(self, target_name, df):
        """Get relevant features for a target based on target_pairs."""
        target_info = self.target_pairs[self.target_pairs['target'] == target_name].iloc[0]
        pair = target_info['pair']
        
        relevant_features = []
        
        if ' - ' in pair:
            # Spread target
            asset1, asset2 = pair.split(' - ')
            relevant_features.extend([col for col in df.columns if asset1 in col])
            relevant_features.extend([col for col in df.columns if asset2 in col])
        else:
            # Single asset target
            relevant_features.extend([col for col in df.columns if pair in col])
        
        # Add general features
        relevant_features.extend([col for col in df.columns if 'FX_' in col][:5])
        relevant_features.extend([col for col in df.columns if 'LME_' in col][:5])
        
        # Remove duplicates and date_id
        relevant_features = list(set(relevant_features))
        if 'date_id' in relevant_features:
            relevant_features.remove('date_id')
        
        return relevant_features
    
    def train_models(self, train_data, train_labels):
        """Train models for all targets."""
        print("🏋️ Training models for all targets...")
        
        # Create features
        engineered_data = create_features(train_data)
        
        # Train for each target
        target_columns = [col for col in train_labels.columns if col.startswith('target_')]
        
        for i, target_name in enumerate(target_columns):
            if i % 50 == 0:
                print(f"  Training target {i+1}/{len(target_columns)}")
            
            try:
                # Get target values
                target_values = train_labels[target_name].values
                
                # Get relevant features
                relevant_features = self.get_target_features(target_name, engineered_data)
                
                if len(relevant_features) == 0:
                    # Use subset of features if none found
                    relevant_features = [col for col in engineered_data.columns if col != 'date_id'][:30]
                
                # Prepare data
                X = engineered_data[relevant_features].values
                y = target_values
                
                # Remove NaN values
                valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                X = X[valid_mask]
                y = y[valid_mask]
                
                if len(X) > 50:  # Only train if enough data
                    # Create ensemble
                    models = [
                        Ridge(alpha=1.0, random_state=42),
                        RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42),
                        xgb.XGBRegressor(n_estimators=50, max_depth=6, learning_rate=0.1, random_state=42)
                    ]
                    
                    # Train models
                    trained_models = []
                    for model in models:
                        model.fit(X, y)
                        trained_models.append(model)
                    
                    # Store models and features
                    self.models[target_name] = trained_models
                    self.features[target_name] = relevant_features
                else:
                    self.models[target_name] = None
                    
            except Exception as e:
                print(f"⚠️ Error training {target_name}: {str(e)}")
                self.models[target_name] = None
        
        print(f"✅ Training completed for {len(self.models)} targets")
    
    def predict_target(self, target_name, test_data):
        """Predict for a specific target."""
        if target_name not in self.models or self.models[target_name] is None:
            # Return conservative prediction
            return np.random.normal(0, 0.005)
        
        models = self.models[target_name]
        features = self.features[target_name]
        
        # Get available features
        available_features = [f for f in features if f in test_data.columns]
        
        if len(available_features) == 0:
            return np.random.normal(0, 0.005)
        
        X = test_data[available_features].values
        
        # Handle NaN values
        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0)
        
        # Make ensemble prediction
        predictions = []
        for model in models:
            pred = model.predict(X.reshape(1, -1))[0]
            predictions.append(pred)
        
        return np.mean(predictions)

# ============================================================================
# MAIN PREDICTION AND SUBMISSION GENERATION
# ============================================================================

# Initialize predictor
print("🔧 Initializing target predictor...")
predictor = TargetPredictor(target_pairs)

# Train models
print("📚 Training models on historical data...")
predictor.train_models(train_data, train_labels)

# Create features for test data
print("🔧 Creating features for test data...")
engineered_test_data = create_features(test_data)

# Generate predictions for all test samples
print("🔮 Generating predictions for all test samples...")
all_predictions = []

for idx, test_row in engineered_test_data.iterrows():
    if idx % 50 == 0:
        print(f"  Predicting sample {idx+1}/{len(engineered_test_data)}")
    
    # Create predictions for this sample
    sample_predictions = {'date_id': test_row['date_id']}
    
    # Predict for all 424 targets
    for i in range(424):
        target_name = f'target_{i}'
        try:
            pred_value = predictor.predict_target(target_name, pd.DataFrame([test_row]))
            sample_predictions[target_name] = pred_value
        except Exception as e:
            # Fallback to conservative prediction
            sample_predictions[target_name] = np.random.normal(0, 0.005)
    
    all_predictions.append(sample_predictions)

# Create submission DataFrame
print("📊 Creating submission DataFrame...")
submission_df = pd.DataFrame(all_predictions)

# Ensure all target columns are present
expected_targets = [f'target_{i}' for i in range(424)]
for target in expected_targets:
    if target not in submission_df.columns:
        submission_df[target] = np.random.normal(0, 0.005)

# Ensure date_id is the first column
cols = ['date_id'] + [col for col in submission_df.columns if col != 'date_id']
submission_df = submission_df[cols]

# Clean up predictions (remove infinite values, etc.)
for col in submission_df.columns:
    if col.startswith('target_'):
        submission_df[col] = submission_df[col].replace([np.inf, -np.inf], 0)
        submission_df[col] = submission_df[col].fillna(0)

print(f"✅ Submission DataFrame created: {submission_df.shape}")

# ============================================================================
# SAVE SUBMISSION FILE
# ============================================================================

# Save as parquet file (required by competition)
print("💾 Saving submission.parquet...")
submission_df.to_parquet('submission.parquet', index=False)

# Also save as CSV for verification
print("💾 Saving submission.csv for verification...")
submission_df.to_csv('submission.csv', index=False)

# Display summary
print("\n" + "="*60)
print("📊 SUBMISSION SUMMARY")
print("="*60)
print(f"📅 Date range: {submission_df['date_id'].min()} to {submission_df['date_id'].max()}")
print(f"🎯 Target columns: {len([col for col in submission_df.columns if col.startswith('target_')])}")
print(f"📈 Sample predictions range: {submission_df[[col for col in submission_df.columns if col.startswith('target_')]].min().min():.6f} to {submission_df[[col for col in submission_df.columns if col.startswith('target_')]].max().max():.6f}")
print(f"💾 Files saved:")
print(f"   - submission.parquet (required by competition)")
print(f"   - submission.csv (for verification)")
print("="*60)

print("🎉 Submission file generated successfully!")
print(f"⏰ Completed at: {datetime.now()}")

# Display first few rows for verification
print("\n📋 First few rows of submission:")
print(submission_df.head(3))

print("\n🚀 Ready to submit! The submission.parquet file has been created.") 