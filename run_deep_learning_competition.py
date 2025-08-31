#!/usr/bin/env python3
"""
MITSUI&CO. Commodity Prediction Challenge - Deep Learning Pipeline
This script implements deep learning models (LSTM, GRU, Transformer, Hybrid) for the competition.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import json

# Add src to path
sys.path.append('src')

# Import custom modules
from data_processing import load_data, add_lag_features, add_rolling_features
from deep_learning_models import (
    get_deep_learning_model, 
    get_trainer, 
    PyTorchTrainer, 
    KerasTrainer
)
from cv import time_series_cv_split
from ensemble import simple_average

warnings.filterwarnings('ignore')

class DeepLearningCompetitionPipeline:
    """Deep Learning Pipeline for MITSUI Commodity Prediction Challenge"""
    
    def __init__(self, data_dir='data', output_dir='outputs'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            'lstm_pytorch': {
                'hidden_size': 256,
                'num_layers': 3,
                'dropout': 0.3
            },
            'gru_pytorch': {
                'hidden_size': 256,
                'num_layers': 3,
                'dropout': 0.3
            },
            'transformer_pytorch': {
                'd_model': 256,
                'nhead': 8,
                'num_layers': 4,
                'dropout': 0.1
            },
            'hybrid_pytorch': {
                'hidden_size': 256,
                'num_layers': 3,
                'dropout': 0.3
            },
            'lstm_keras': {
                'units': 256,
                'layers': 3,
                'dropout': 0.3
            },
            'gru_keras': {
                'units': 256,
                'layers': 3,
                'dropout': 0.3
            },
            'transformer_keras': {
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 4,
                'dropout': 0.1
            }
        }
        
        # Training parameters
        self.training_params = {
            'sequence_length': 20,
            'batch_size': 64,
            'epochs': 200,
            'patience': 30,
            'learning_rate': 0.001
        }
        
        # Data
        self.train_data = None
        self.test_data = None
        self.train_labels = None
        self.target_pairs = None
        
        # Models and predictions
        self.models = {}
        self.predictions = {}
        self.ensemble_predictions = None
        
    def load_competition_data(self):
        """Load competition data files"""
        print("Loading competition data...")
        
        # Load main data files
        self.train_data = pd.read_csv(self.data_dir / 'train.csv')
        self.test_data = pd.read_csv(self.data_dir / 'test.csv')
        self.train_labels = pd.read_csv(self.data_dir / 'train_labels.csv')
        self.target_pairs = pd.read_csv(self.data_dir / 'target_pairs.csv')
        
        print(f"Train data shape: {self.train_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")
        print(f"Train labels shape: {self.train_labels.shape}")
        print(f"Target pairs: {len(self.target_pairs)}")
        
        # Basic preprocessing
        self.train_data['timestamp'] = pd.to_datetime(self.train_data['timestamp'])
        self.test_data['timestamp'] = pd.to_datetime(self.test_data['timestamp'])
        
        # Sort by timestamp
        self.train_data = self.train_data.sort_values('timestamp').reset_index(drop=True)
        self.test_data = self.test_data.sort_values('timestamp').reset_index(drop=True)
        
        print("Data loading completed!")
        
    def prepare_features(self):
        """Prepare features for deep learning models"""
        print("Preparing features...")
        
        # Combine train and test for feature engineering
        combined_data = pd.concat([
            self.train_data.drop('timestamp', axis=1),
            self.test_data.drop('timestamp', axis=1)
        ], axis=0).reset_index(drop=True)
        
        # Add lag features
        combined_data = add_lag_features(combined_data, lags=[1, 2, 3, 5, 10])
        
        # Add rolling features
        combined_data = add_rolling_features(combined_data, windows=[5, 10, 20])
        
        # Handle missing values
        combined_data = combined_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Split back to train and test
        train_size = len(self.train_data)
        self.train_features = combined_data.iloc[:train_size].values
        self.test_features = combined_data.iloc[train_size:].values
        
        print(f"Train features shape: {self.train_features.shape}")
        print(f"Test features shape: {self.test_features.shape}")
        print("Feature preparation completed!")
        
    def prepare_targets(self):
        """Prepare target variables"""
        print("Preparing targets...")
        
        # Get target columns (excluding timestamp and id)
        target_cols = [col for col in self.train_labels.columns if col not in ['timestamp', 'id']]
        self.target_columns = target_cols
        
        # Prepare target values
        self.target_values = self.train_labels[target_cols].values
        
        print(f"Number of targets: {len(self.target_columns)}")
        print(f"Target values shape: {self.target_values.shape}")
        print("Target preparation completed!")
        
    def train_deep_learning_models(self):
        """Train deep learning models for each target"""
        print("Training deep learning models...")
        
        # Select a subset of targets for demonstration (first 10)
        num_targets = min(10, len(self.target_columns))
        selected_targets = self.target_columns[:num_targets]
        
        print(f"Training models for {num_targets} targets...")
        
        for target_idx, target_name in enumerate(selected_targets):
            print(f"\nTraining models for target {target_idx + 1}/{num_targets}: {target_name}")
            
            # Get target values
            y = self.target_values[:, target_idx]
            
            # Remove NaN targets
            valid_indices = ~np.isnan(y)
            X_valid = self.train_features[valid_indices]
            y_valid = y[valid_indices]
            
            if len(X_valid) < self.training_params['sequence_length'] + 100:
                print(f"Skipping {target_name} - insufficient data")
                continue
                
            # Time series split
            train_indices, val_indices = time_series_cv_split(
                len(X_valid), 
                n_splits=1, 
                test_size=0.2
            )
            
            X_train = X_valid[train_indices]
            y_train = y_valid[train_indices]
            X_val = X_valid[val_indices]
            y_val = y_valid[val_indices]
            
            # Train different model types
            for model_type, config in self.model_configs.items():
                try:
                    print(f"  Training {model_type}...")
                    
                    # Get model
                    if 'pytorch' in model_type:
                        model = get_deep_learning_model(model_type, X_train.shape[1], **config)
                        trainer = PyTorchTrainer(model)
                        
                        # Prepare data
                        train_loader, scaler = trainer.prepare_data(
                            X_train, y_train, 
                            self.training_params['sequence_length'],
                            self.training_params['batch_size']
                        )
                        
                        # Train model
                        train_losses, val_losses = trainer.train(
                            train_loader, 
                            epochs=self.training_params['epochs'],
                            lr=self.training_params['learning_rate'],
                            patience=self.training_params['patience']
                        )
                        
                        # Make predictions
                        train_pred = trainer.predict(X_train, self.training_params['sequence_length'])
                        val_pred = trainer.predict(X_val, self.training_params['sequence_length'])
                        
                        # Store model and predictions
                        model_key = f"{target_name}_{model_type}"
                        self.models[model_key] = {
                            'model': trainer,
                            'scaler': scaler,
                            'train_pred': train_pred,
                            'val_pred': val_pred,
                            'val_actual': y_val[self.training_params['sequence_length']:],
                            'train_losses': train_losses,
                            'val_losses': val_losses
                        }
                        
                        # Calculate metrics
                        val_rmse = np.sqrt(np.mean((val_pred - y_val[self.training_params['sequence_length']:])**2))
                        print(f"    {model_type} - Val RMSE: {val_rmse:.6f}")
                        
                    elif 'keras' in model_type:
                        model = get_deep_learning_model(model_type, X_train.shape[1], **config)
                        trainer = KerasTrainer(model)
                        
                        # Prepare data
                        X_train_seq, y_train_seq, scaler = trainer.prepare_data(
                            X_train, y_train, self.training_params['sequence_length']
                        )
                        X_val_seq, y_val_seq, _ = trainer.prepare_data(
                            X_val, y_val, self.training_params['sequence_length']
                        )
                        
                        # Train model
                        history = trainer.train(
                            X_train_seq, y_train_seq,
                            X_val_seq, y_val_seq,
                            epochs=self.training_params['epochs'],
                            batch_size=self.training_params['batch_size'],
                            patience=self.training_params['patience']
                        )
                        
                        # Make predictions
                        train_pred = trainer.predict(X_train, self.training_params['sequence_length'])
                        val_pred = trainer.predict(X_val, self.training_params['sequence_length'])
                        
                        # Store model and predictions
                        model_key = f"{target_name}_{model_type}"
                        self.models[model_key] = {
                            'model': trainer,
                            'scaler': scaler,
                            'train_pred': train_pred,
                            'val_pred': val_pred,
                            'val_actual': y_val[self.training_params['sequence_length']:],
                            'history': history
                        }
                        
                        # Calculate metrics
                        val_rmse = np.sqrt(np.mean((val_pred - y_val[self.training_params['sequence_length']:])**2))
                        print(f"    {model_type} - Val RMSE: {val_rmse:.6f}")
                        
                except Exception as e:
                    print(f"    Error training {model_type}: {str(e)}")
                    continue
        
        print(f"\nTraining completed! Trained {len(self.models)} models.")
        
    def generate_predictions(self):
        """Generate predictions for test data"""
        print("Generating predictions...")
        
        # Generate predictions for each trained model
        for model_key, model_info in self.models.items():
            try:
                target_name, model_type = model_key.split('_', 1)
                trainer = model_info['model']
                
                print(f"Generating predictions for {model_key}...")
                
                # Make predictions on test data
                test_pred = trainer.predict(self.test_features, self.training_params['sequence_length'])
                
                # Store predictions
                if target_name not in self.predictions:
                    self.predictions[target_name] = {}
                self.predictions[target_name][model_type] = test_pred
                
            except Exception as e:
                print(f"Error generating predictions for {model_key}: {str(e)}")
                continue
        
        print("Prediction generation completed!")
        
    def create_ensemble_predictions(self):
        """Create ensemble predictions"""
        print("Creating ensemble predictions...")
        
        # For each target, ensemble predictions from different models
        for target_name in self.predictions:
            model_predictions = []
            
            for model_type, pred in self.predictions[target_name].items():
                model_predictions.append(pred)
            
            if len(model_predictions) > 1:
                # Simple averaging
                ensemble_pred = np.mean(model_predictions, axis=0)
                self.predictions[target_name]['ensemble'] = ensemble_pred
                print(f"Created ensemble for {target_name} using {len(model_predictions)} models")
        
        print("Ensemble creation completed!")
        
    def create_submission_file(self):
        """Create competition submission file"""
        print("Creating submission file...")
        
        # Create submission dataframe
        submission_data = []
        
        for target_name in self.predictions:
            # Use ensemble prediction if available, otherwise use first available model
            if 'ensemble' in self.predictions[target_name]:
                pred = self.predictions[target_name]['ensemble']
            else:
                pred = list(self.predictions[target_name].values())[0]
            
            # Create rows for each prediction
            for i, pred_value in enumerate(pred):
                submission_data.append({
                    'id': f"{target_name}_{i}",
                    'prediction': pred_value
                })
        
        # Create submission dataframe
        submission_df = pd.DataFrame(submission_data)
        
        # Save submission file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_file = self.output_dir / f"deep_learning_submission_{timestamp}.csv"
        submission_df.to_csv(submission_file, index=False)
        
        print(f"Submission file saved: {submission_file}")
        print(f"Submission shape: {submission_df.shape}")
        
        return submission_file
        
    def save_model_artifacts(self):
        """Save trained models and artifacts"""
        print("Saving model artifacts...")
        
        # Create models directory
        models_dir = self.output_dir / 'deep_learning_models'
        models_dir.mkdir(exist_ok=True)
        
        # Save model information
        model_info = {}
        for model_key, model_info_dict in self.models.items():
            target_name, model_type = model_key.split('_', 1)
            
            if target_name not in model_info:
                model_info[target_name] = {}
            
            # Extract key metrics
            if 'val_losses' in model_info_dict:
                best_val_loss = min(model_info_dict['val_losses'])
                model_info[target_name][model_type] = {
                    'best_val_loss': best_val_loss,
                    'final_train_loss': model_info_dict['train_losses'][-1] if model_info_dict['train_losses'] else None
                }
            elif 'history' in model_info_dict:
                val_losses = model_info_dict['history'].history.get('val_loss', [])
                train_losses = model_info_dict['history'].history.get('loss', [])
                model_info[target_name][model_type] = {
                    'best_val_loss': min(val_losses) if val_losses else None,
                    'final_train_loss': train_losses[-1] if train_losses else None
                }
        
        # Save model info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        info_file = models_dir / f"model_info_{timestamp}.json"
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
        
        print(f"Model artifacts saved to: {models_dir}")
        
    def run_pipeline(self):
        """Run the complete deep learning pipeline"""
        print("=" * 60)
        print("MITSUI&CO. Commodity Prediction Challenge - Deep Learning Pipeline")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Step 1: Load data
            self.load_competition_data()
            
            # Step 2: Prepare features
            self.prepare_features()
            
            # Step 3: Prepare targets
            self.prepare_targets()
            
            # Step 4: Train models
            self.train_deep_learning_models()
            
            # Step 5: Generate predictions
            self.generate_predictions()
            
            # Step 6: Create ensemble
            self.create_ensemble_predictions()
            
            # Step 7: Create submission
            submission_file = self.create_submission_file()
            
            # Step 8: Save artifacts
            self.save_model_artifacts()
            
            total_time = time.time() - start_time
            print(f"\nPipeline completed successfully in {total_time:.2f} seconds!")
            print(f"Submission file: {submission_file}")
            
        except Exception as e:
            print(f"Pipeline failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
        return True

def main():
    """Main function"""
    # Create pipeline
    pipeline = DeepLearningCompetitionPipeline()
    
    # Run pipeline
    success = pipeline.run_pipeline()
    
    if success:
        print("\nDeep Learning Pipeline completed successfully!")
        print("Check the outputs/ directory for results.")
    else:
        print("\nDeep Learning Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
