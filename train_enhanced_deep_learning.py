#!/usr/bin/env python3
"""
Enhanced Deep Learning Training for MITSUI Commodity Prediction Challenge
This script includes comprehensive evaluation and analysis capabilities.
"""

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

# Import working modules
from deep_learning_models import LSTMModel, GRUModel, PyTorchTrainer
from data_processing import add_lag_features, add_rolling_features
from deep_learning_evaluation import DeepLearningEvaluator, evaluate_model_performance

warnings.filterwarnings('ignore')

class EnhancedDeepLearningPipeline:
    """Enhanced Deep Learning Pipeline with comprehensive evaluation"""
    
    def __init__(self, data_dir='data', output_dir='outputs'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create evaluation directory
        self.eval_dir = self.output_dir / 'evaluation_results'
        self.eval_dir.mkdir(exist_ok=True)
        
        # Model configurations for working models
        self.model_configs = {
            'lstm': {
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.2
            },
            'gru': {
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.2
            }
        }
        
        # Training parameters
        self.training_params = {
            'sequence_length': 15,
            'batch_size': 32,
            'epochs': 50,
            'patience': 15,
            'learning_rate': 0.001
        }
        
        # Data
        self.train_data = None
        self.test_data = None
        self.train_labels = None
        
        # Models and predictions
        self.models = {}
        self.predictions = {}
        
        # Evaluation
        self.evaluator = DeepLearningEvaluator()
        
    def load_data(self):
        """Load competition data"""
        print("Loading competition data...")
        
        try:
            self.train_data = pd.read_csv(self.data_dir / 'train.csv')
            self.test_data = pd.read_csv(self.data_dir / 'test.csv')
            self.train_labels = pd.read_csv(self.data_dir / 'train_labels.csv')
            
            print(f"Train data shape: {self.train_data.shape}")
            print(f"Test data shape: {self.test_data.shape}")
            print(f"Train labels shape: {self.train_labels.shape}")
            
            # Basic preprocessing - check if timestamp column exists
            if 'timestamp' in self.train_data.columns:
                self.train_data['timestamp'] = pd.to_datetime(self.train_data['timestamp'])
                self.test_data['timestamp'] = pd.to_datetime(self.test_data['timestamp'])
                
                # Sort by timestamp
                self.train_data = self.train_data.sort_values('timestamp').reset_index(drop=True)
                self.test_data = self.test_data.sort_values('timestamp').reset_index(drop=True)
            else:
                print("  No timestamp column found, using existing order")
            
            print("✓ Data loading completed!")
            
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            raise
    
    def prepare_features(self):
        """Prepare features for deep learning"""
        print("Preparing features...")
        
        try:
            # Combine train and test for feature engineering
            # Handle columns that might not exist in both datasets
            train_cols = set(self.train_data.columns)
            test_cols = set(self.test_data.columns)
            common_cols = train_cols.intersection(test_cols)
            
            print(f"  Common columns between train and test: {len(common_cols)}")
            
            combined_data = pd.concat([
                self.train_data[list(common_cols)],
                self.test_data[list(common_cols)]
            ], axis=0).reset_index(drop=True)
            
            # Add lag features - use first 10 columns for demonstration
            feature_cols = list(common_cols)[:10]  # Use first 10 columns to avoid too many features
            combined_data = add_lag_features(combined_data, cols=feature_cols, lags=[1, 2, 3, 5])
            
            # Add rolling features
            combined_data = add_rolling_features(combined_data, cols=feature_cols, windows=[5, 10])
            
            # Handle missing values
            combined_data = combined_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Split back to train and test
            train_size = len(self.train_data)
            self.train_features = combined_data.iloc[:train_size].values
            self.test_features = combined_data.iloc[train_size:].values
            
            print(f"Train features shape: {self.train_features.shape}")
            print(f"Test features shape: {self.test_features.shape}")
            print("✓ Feature preparation completed!")
            
        except Exception as e:
            print(f"✗ Error preparing features: {str(e)}")
            raise
    
    def prepare_targets(self):
        """Prepare target variables"""
        print("Preparing targets...")
        
        try:
            # Get target columns (excluding timestamp and id)
            target_cols = [col for col in self.train_labels.columns if col not in ['timestamp', 'id']]
            self.target_columns = target_cols
            
            # Prepare target values
            self.target_values = self.train_labels[target_cols].values
            
            print(f"Number of targets: {len(self.target_columns)}")
            print(f"Target values shape: {self.target_values.shape}")
            print("✓ Target preparation completed!")
            
        except Exception as e:
            print(f"✗ Error preparing targets: {str(e)}")
            raise
    
    def train_models_for_target(self, target_name, target_values):
        """Train models for a specific target with enhanced evaluation"""
        print(f"  Training models for target: {target_name}")
        
        # Remove NaN targets
        valid_indices = ~np.isnan(target_values)
        X_valid = self.train_features[valid_indices]
        y_valid = target_values[valid_indices]
        
        if len(X_valid) < self.training_params['sequence_length'] + 50:
            print(f"    Skipping {target_name} - insufficient data")
            return None
        
        # Simple train/validation split (last 20% for validation)
        split_idx = int(len(X_valid) * 0.8)
        X_train = X_valid[:split_idx]
        y_train = y_valid[:split_idx]
        X_val = X_valid[split_idx:]
        y_val = y_valid[split_idx:]
        
        target_models = {}
        
        # Train different model types
        for model_name, config in self.model_configs.items():
            try:
                print(f"    Training {model_name}...")
                
                # Create model
                if model_name == 'lstm':
                    model = LSTMModel(X_train.shape[1], **config)
                elif model_name == 'gru':
                    model = GRUModel(X_train.shape[1], **config)
                else:
                    continue
                
                # Create trainer
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
                val_pred = trainer.predict(X_val, self.training_params['sequence_length'])
                
                # Calculate metrics
                val_actual = y_val[self.training_params['sequence_length']:]
                if len(val_pred) == len(val_actual):
                    # Enhanced evaluation
                    _, basic_metrics, competition_metrics = evaluate_model_performance(
                        val_actual, val_pred, target_name, model_name, self.evaluator
                    )
                    
                    # Store model and predictions
                    target_models[model_name] = {
                        'trainer': trainer,
                        'scaler': scaler,
                        'val_pred': val_pred,
                        'val_actual': val_actual,
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'basic_metrics': basic_metrics,
                        'competition_metrics': competition_metrics
                    }
                    
                    # Create evaluation plots
                    self.create_evaluation_plots(target_name, model_name, val_actual, val_pred, 
                                              train_losses, val_losses)
                
            except Exception as e:
                print(f"      Error training {model_name}: {str(e)}")
                continue
        
        return target_models
    
    def create_evaluation_plots(self, target_name, model_name, y_true, y_pred, train_losses, val_losses):
        """Create comprehensive evaluation plots"""
        try:
            # Create plots directory
            plots_dir = self.eval_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            # Predictions vs actual plot
            pred_plot_path = plots_dir / f"{target_name}_{model_name}_predictions.png"
            self.evaluator.plot_predictions_vs_actual(
                y_true, y_pred, target_name, model_name, str(pred_plot_path)
            )
            
            # Training curves plot
            train_plot_path = plots_dir / f"{target_name}_{model_name}_training.png"
            self.evaluator.plot_training_curves(
                train_losses, val_losses, target_name, model_name, str(train_plot_path)
            )
            
            print(f"      ✓ Evaluation plots saved for {model_name}")
            
        except Exception as e:
            print(f"      ✗ Error creating plots for {model_name}: {str(e)}")
    
    def train_all_models(self):
        """Train models for all targets"""
        print("Training deep learning models...")
        
        # Select a subset of targets for demonstration (first 5)
        num_targets = min(5, len(self.target_columns))
        selected_targets = self.target_columns[:num_targets]
        
        print(f"Training models for {num_targets} targets...")
        
        for target_idx, target_name in enumerate(selected_targets):
            print(f"\nTarget {target_idx + 1}/{num_targets}: {target_name}")
            
            # Get target values
            y = self.target_values[:, target_idx]
            
            # Train models for this target
            target_models = self.train_models_for_target(target_name, y)
            
            if target_models:
                self.models[target_name] = target_models
                print(f"  ✓ Trained {len(target_models)} models for {target_name}")
            else:
                print(f"  ✗ No models trained for {target_name}")
        
        print(f"\n✓ Training completed! Trained models for {len(self.models)} targets.")
    
    def generate_predictions(self):
        """Generate predictions for test data"""
        print("Generating predictions...")
        
        try:
            for target_name, target_models in self.models.items():
                print(f"  Generating predictions for {target_name}...")
                
                target_predictions = {}
                
                for model_name, model_info in target_models.items():
                    try:
                        trainer = model_info['trainer']
                        
                        # Make predictions on test data
                        test_pred = trainer.predict(self.test_features, self.training_params['sequence_length'])
                        target_predictions[model_name] = test_pred
                        
                        print(f"    {model_name} predictions shape: {test_pred.shape}")
                        
                    except Exception as e:
                        print(f"    Error generating predictions for {model_name}: {str(e)}")
                        continue
                
                if target_predictions:
                    self.predictions[target_name] = target_predictions
                    print(f"  ✓ Generated predictions for {target_name}")
                
            print("✓ Prediction generation completed!")
            
        except Exception as e:
            print(f"✗ Error generating predictions: {str(e)}")
            raise
    
    def create_ensemble_predictions(self):
        """Create ensemble predictions"""
        print("Creating ensemble predictions...")
        
        try:
            for target_name in self.predictions:
                model_predictions = []
                
                for model_name, pred in self.predictions[target_name].items():
                    model_predictions.append(pred)
                
                if len(model_predictions) > 1:
                    # Simple averaging
                    ensemble_pred = np.mean(model_predictions, axis=0)
                    self.predictions[target_name]['ensemble'] = ensemble_pred
                    print(f"  Created ensemble for {target_name} using {len(model_predictions)} models")
            
            print("✓ Ensemble creation completed!")
            
        except Exception as e:
            print(f"✗ Error creating ensemble: {str(e)}")
            raise
    
    def create_submission_file(self):
        """Create competition submission file"""
        print("Creating submission file...")
        
        try:
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
            submission_file = self.output_dir / f"enhanced_deep_learning_submission_{timestamp}.csv"
            submission_df.to_csv(submission_file, index=False)
            
            print(f"✓ Submission file saved: {submission_file}")
            print(f"  Submission shape: {submission_df.shape}")
            
            return submission_file
            
        except Exception as e:
            print(f"✗ Error creating submission: {str(e)}")
            raise
    
    def save_comprehensive_results(self):
        """Save comprehensive results including models, predictions, and evaluation"""
        print("Saving comprehensive results...")
        
        try:
            # Save model information
            model_info = {}
            for target_name, target_models in self.models.items():
                model_info[target_name] = {}
                
                for model_name, model_info_dict in target_models.items():
                    model_info[target_name][model_name] = {
                        'basic_metrics': model_info_dict.get('basic_metrics', {}),
                        'competition_metrics': model_info_dict.get('competition_metrics', {}),
                        'final_train_loss': model_info_dict.get('train_losses', [])[-1] if model_info_dict.get('train_losses') else None,
                        'best_val_loss': min(model_info_dict.get('val_losses', [])) if model_info_dict.get('val_losses') else None
                    }
            
            # Save model info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            info_file = self.output_dir / f"enhanced_model_info_{timestamp}.json"
            with open(info_file, 'w') as f:
                json.dump(model_info, f, indent=2, default=str)
            
            # Save evaluation results
            eval_results_dir = self.eval_dir / f"evaluation_{timestamp}"
            self.evaluator.save_evaluation_results(eval_results_dir)
            
            print(f"✓ Comprehensive results saved to: {self.output_dir}")
            print(f"  Model info: {info_file}")
            print(f"  Evaluation results: {eval_results_dir}")
            
        except Exception as e:
            print(f"✗ Error saving comprehensive results: {str(e)}")
            raise
    
    def run_pipeline(self):
        """Run the complete enhanced pipeline"""
        print("=" * 70)
        print("MITSUI&CO. Commodity Prediction Challenge - Enhanced Deep Learning Pipeline")
        print("=" * 70)
        
        start_time = time.time()
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Prepare features
            self.prepare_features()
            
            # Step 3: Prepare targets
            self.prepare_targets()
            
            # Step 4: Train models with enhanced evaluation
            self.train_all_models()
            
            # Step 5: Generate predictions
            self.generate_predictions()
            
            # Step 6: Create ensemble
            self.create_ensemble_predictions()
            
            # Step 7: Create submission
            submission_file = self.create_submission_file()
            
            # Step 8: Save comprehensive results
            self.save_comprehensive_results()
            
            total_time = time.time() - start_time
            print(f"\n🎉 Enhanced Pipeline completed successfully in {total_time:.2f} seconds!")
            print(f"📁 Submission file: {submission_file}")
            print(f"📊 Evaluation results: {self.eval_dir}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Enhanced Pipeline failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function"""
    # Create enhanced pipeline
    pipeline = EnhancedDeepLearningPipeline()
    
    # Run pipeline
    success = pipeline.run_pipeline()
    
    if success:
        print("\n✅ Enhanced Deep Learning Pipeline completed successfully!")
        print("📂 Check the outputs/ and evaluation_results/ directories for comprehensive results.")
    else:
        print("\n❌ Enhanced Deep Learning Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
