#!/usr/bin/env python3
"""
Comprehensive Deep Learning Training for MITSUI Commodity Prediction Challenge
This script integrates all microclusters: optimization, advanced techniques, and testing.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import json
import optuna

# Add src to path
sys.path.append('src')

# Import all modules
from deep_learning_models import LSTMModel, GRUModel, TransformerModel, HybridModel, PyTorchTrainer
from advanced_deep_learning import MultiTaskLSTM, AttentionLSTM, TemporalFusionTransformer, MultiTaskTrainer
from deep_learning_evaluation import DeepLearningEvaluator, evaluate_model_performance
from deep_learning_optimization import DeepLearningOptimizer, optimize_multiple_models
from deep_learning_testing import DeepLearningTestSuite, run_quick_validation
from data_processing import add_lag_features, add_rolling_features

warnings.filterwarnings('ignore')

class ComprehensiveDeepLearningPipeline:
    """Comprehensive Deep Learning Pipeline integrating all microclusters"""
    
    def __init__(self, data_dir='data', output_dir='outputs'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create specialized directories
        self.eval_dir = self.output_dir / 'evaluation_results'
        self.opt_dir = self.output_dir / 'optimization_results'
        self.test_dir = self.output_dir / 'testing_results'
        self.advanced_dir = self.output_dir / 'advanced_models'
        
        for dir_path in [self.eval_dir, self.opt_dir, self.test_dir, self.advanced_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Model configurations for different complexity levels
        self.basic_models = {
            'lstm': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2},
            'gru': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2}
        }
        
        self.advanced_models = {
            'transformer': {'d_model': 128, 'nhead': 8, 'num_layers': 4, 'dropout': 0.1},
            'hybrid': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2},
            'attention_lstm': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2},
            'temporal_fusion': {'d_model': 128, 'nhead': 8, 'num_layers': 4, 'dropout': 0.1}
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
        self.optimized_models = {}
        
        # Evaluation and testing
        self.evaluator = DeepLearningEvaluator()
        self.test_suite = DeepLearningTestSuite()
        
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
            
            # Basic preprocessing
            if 'timestamp' in self.train_data.columns:
                self.train_data['timestamp'] = pd.to_datetime(self.train_data['timestamp'])
                self.test_data['timestamp'] = pd.to_datetime(self.test_data['timestamp'])
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
            train_cols = set(self.train_data.columns)
            test_cols = set(self.test_data.columns)
            common_cols = train_cols.intersection(test_cols)
            
            print(f"  Common columns between train and test: {len(common_cols)}")
            
            combined_data = pd.concat([
                self.train_data[list(common_cols)],
                self.test_data[list(common_cols)]
            ], axis=0).reset_index(drop=True)
            
            # Add lag features
            feature_cols = list(common_cols)[:15]  # Use more features for advanced models
            combined_data = add_lag_features(combined_data, cols=feature_cols, lags=[1, 2, 3, 5, 7])
            
            # Add rolling features
            combined_data = add_rolling_features(combined_data, cols=feature_cols, windows=[5, 10, 15])
            
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
            # Get target columns
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
    
    def run_model_validation(self):
        """Run comprehensive model validation and testing"""
        print("\nRunning model validation and testing...")
        print("=" * 50)
        
        try:
            # Quick validation
            print("1. Quick Model Validation...")
            validation_results = run_quick_validation()
            
            # Comprehensive testing
            print("\n2. Comprehensive Model Testing...")
            test_results = self.test_suite.run_comprehensive_tests(input_size=self.train_features.shape[1])
            
            # Save results
            validation_file = self.test_dir / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(validation_file, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
            
            test_file = self.test_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(test_file, 'w') as f:
                json.dump(test_results, f, indent=2, default=str)
            
            print("✓ Model validation and testing completed!")
            return validation_results, test_results
            
        except Exception as e:
            print(f"✗ Error in model validation: {str(e)}")
            return None, None
    
    def run_hyperparameter_optimization(self):
        """Run hyperparameter optimization for basic models"""
        print("\nRunning hyperparameter optimization...")
        print("=" * 50)
        
        try:
            # Select subset of targets for optimization
            num_targets = min(3, len(self.target_columns))
            selected_targets = self.target_columns[:num_targets]
            
            print(f"Optimizing for {num_targets} targets: {selected_targets}")
            
            # Prepare data for first target
            target_idx = 0
            y = self.target_values[:, target_idx]
            
            # Remove NaN targets
            valid_indices = ~np.isnan(y)
            X_valid = self.train_features[valid_indices]
            y_valid = y[valid_indices]
            
            # Split data
            split_idx = int(len(X_valid) * 0.8)
            X_train = X_valid[:split_idx]
            y_train = y_valid[:split_idx]
            X_val = X_valid[split_idx:]
            y_val = y_valid[split_idx:]
            
            # Run optimization
            optimizer = DeepLearningOptimizer(n_trials=20, timeout=1800)  # 30 minutes timeout
            
            optimization_results = optimizer.optimize_all_models(
                X_train, y_train, X_val, y_val,
                input_size=X_train.shape[1],
                model_types=['lstm', 'gru'],
                max_epochs=30,
                patience=10
            )
            
            # Save optimization results
            opt_file = self.opt_dir / f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            optimizer.save_optimization_results(str(self.opt_dir))
            
            print("✓ Hyperparameter optimization completed!")
            return optimizer, optimization_results
            
        except Exception as e:
            print(f"✗ Error in hyperparameter optimization: {str(e)}")
            return None, None
    
    def train_basic_models(self):
        """Train basic models (LSTM, GRU)"""
        print("\nTraining basic models...")
        print("=" * 50)
        
        # Select subset of targets
        num_targets = min(5, len(self.target_columns))
        selected_targets = self.target_columns[:num_targets]
        
        print(f"Training basic models for {num_targets} targets...")
        
        for target_idx, target_name in enumerate(selected_targets):
            print(f"\nTarget {target_idx + 1}/{num_targets}: {target_name}")
            
            # Get target values
            y = self.target_values[:, target_idx]
            
            # Remove NaN targets
            valid_indices = ~np.isnan(y)
            X_valid = self.train_features[valid_indices]
            y_valid = y[valid_indices]
            
            if len(X_valid) < self.training_params['sequence_length'] + 50:
                print(f"    Skipping {target_name} - insufficient data")
                continue
            
            # Split data
            split_idx = int(len(X_valid) * 0.8)
            X_train = X_valid[:split_idx]
            y_train = y_valid[:split_idx]
            X_val = X_valid[split_idx:]
            y_val = y_valid[split_idx:]
            
            target_models = {}
            
            # Train basic models
            for model_name, config in self.basic_models.items():
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
                        
                        # Store model
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
                
                except Exception as e:
                    print(f"      Error training {model_name}: {str(e)}")
                    continue
            
            if target_models:
                self.models[target_name] = target_models
                print(f"  ✓ Trained {len(target_models)} basic models for {target_name}")
        
        print(f"\n✓ Basic model training completed! Trained models for {len(self.models)} targets.")
    
    def train_advanced_models(self):
        """Train advanced models (Transformer, Hybrid, etc.)"""
        print("\nTraining advanced models...")
        print("=" * 50)
        
        # Select subset of targets
        num_targets = min(3, len(self.target_columns))
        selected_targets = self.target_columns[:num_targets]
        
        print(f"Training advanced models for {num_targets} targets...")
        
        for target_idx, target_name in enumerate(selected_targets):
            if target_name not in self.models:
                continue
                
            print(f"\nTarget {target_idx + 1}/{num_targets}: {target_name}")
            
            # Get target values
            y = self.target_values[:, target_idx]
            
            # Remove NaN targets
            valid_indices = ~np.isnan(y)
            X_valid = self.train_features[valid_indices]
            y_valid = y[valid_indices]
            
            if len(X_valid) < self.training_params['sequence_length'] + 50:
                print(f"    Skipping {target_name} - insufficient data")
                continue
            
            # Split data
            split_idx = int(len(X_valid) * 0.8)
            X_train = X_valid[:split_idx]
            y_train = y_valid[:split_idx]
            X_val = X_valid[split_idx:]
            y_val = y_valid[split_idx:]
            
            # Train advanced models
            for model_name, config in self.advanced_models.items():
                try:
                    print(f"    Training {model_name}...")
                    
                    # Create model
                    if model_name == 'transformer':
                        model = TransformerModel(X_train.shape[1], **config)
                    elif model_name == 'hybrid':
                        model = HybridModel(X_train.shape[1], **config)
                    elif model_name == 'attention_lstm':
                        model = AttentionLSTM(X_train.shape[1], **config)
                    elif model_name == 'temporal_fusion':
                        model = TemporalFusionTransformer(X_train.shape[1], **config)
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
                        
                        # Store model
                        if target_name not in self.models:
                            self.models[target_name] = {}
                        
                        self.models[target_name][model_name] = {
                            'trainer': trainer,
                            'scaler': scaler,
                            'val_pred': val_pred,
                            'val_actual': val_actual,
                            'train_losses': train_losses,
                            'val_losses': val_losses,
                            'basic_metrics': basic_metrics,
                            'competition_metrics': competition_metrics
                        }
                
                except Exception as e:
                    print(f"      Error training {model_name}: {str(e)}")
                    continue
        
        print(f"\n✓ Advanced model training completed!")
    
    def train_multi_task_model(self):
        """Train multi-task learning model"""
        print("\nTraining multi-task learning model...")
        print("=" * 50)
        
        try:
            # Select subset of targets for multi-task learning
            num_targets = min(5, len(self.target_columns))
            selected_targets = self.target_columns[:num_targets]
            
            print(f"Training multi-task model for {num_targets} targets...")
            
            # Prepare multi-task data
            y_dict = {}
            valid_indices = None
            
            for i, target_name in enumerate(selected_targets):
                y = self.target_values[:, i]
                
                if valid_indices is None:
                    valid_indices = ~np.isnan(y)
                else:
                    valid_indices = valid_indices & (~np.isnan(y))
                
                y_dict[f'target_{i}'] = y
            
            if valid_indices is None or np.sum(valid_indices) < self.training_params['sequence_length'] + 50:
                print("    Insufficient valid data for multi-task learning")
                return
            
            X_valid = self.train_features[valid_indices]
            y_valid_dict = {k: v[valid_indices] for k, v in y_dict.items()}
            
            # Split data
            split_idx = int(len(X_valid) * 0.8)
            X_train = X_valid[:split_idx]
            y_train_dict = {k: v[:split_idx] for k, v in y_valid_dict.items()}
            X_val = X_valid[split_idx:]
            y_val_dict = {k: v[split_idx:] for k, v in y_valid_dict.items()}
            
            # Create multi-task model
            model = MultiTaskLSTM(
                input_size=X_train.shape[1],
                hidden_size=128,
                num_layers=2,
                dropout=0.2,
                num_targets=len(selected_targets)
            )
            
            # Create trainer
            trainer = MultiTaskTrainer(model)
            
            # Prepare data
            train_loader, scaler = trainer.prepare_data(
                X_train, y_train_dict,
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
            
            # Store multi-task model
            self.models['multi_task'] = {
                'multi_task_lstm': {
                    'trainer': trainer,
                    'scaler': scaler,
                    'val_pred': val_pred,
                    'val_actual': y_val_dict,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'targets': selected_targets
                }
            }
            
            print("✓ Multi-task learning model training completed!")
            
        except Exception as e:
            print(f"✗ Error in multi-task training: {str(e)}")
    
    def generate_predictions(self):
        """Generate predictions for test data"""
        print("\nGenerating predictions...")
        print("=" * 50)
        
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
        print("\nCreating ensemble predictions...")
        print("=" * 50)
        
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
        print("\nCreating submission file...")
        print("=" * 50)
        
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
            submission_file = self.output_dir / f"comprehensive_deep_learning_submission_{timestamp}.csv"
            submission_df.to_csv(submission_file, index=False)
            
            print(f"✓ Submission file saved: {submission_file}")
            print(f"  Submission shape: {submission_df.shape}")
            
            return submission_file
            
        except Exception as e:
            print(f"✗ Error creating submission: {str(e)}")
            raise
    
    def save_comprehensive_results(self):
        """Save comprehensive results"""
        print("\nSaving comprehensive results...")
        print("=" * 50)
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
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
            info_file = self.output_dir / f"comprehensive_model_info_{timestamp}.json"
            with open(info_file, 'w') as f:
                json.dump(model_info, f, indent=2, default=str)
            
            # Save evaluation results
            eval_results_dir = self.eval_dir / f"comprehensive_evaluation_{timestamp}"
            self.evaluator.save_evaluation_results(eval_results_dir)
            
            print(f"✓ Comprehensive results saved to: {self.output_dir}")
            print(f"  Model info: {info_file}")
            print(f"  Evaluation results: {eval_results_dir}")
            
        except Exception as e:
            print(f"✗ Error saving comprehensive results: {str(e)}")
            raise
    
    def run_pipeline(self):
        """Run the complete comprehensive pipeline"""
        print("=" * 80)
        print("MITSUI&CO. Commodity Prediction Challenge - Comprehensive Deep Learning Pipeline")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Prepare features
            self.prepare_features()
            
            # Step 3: Prepare targets
            self.prepare_targets()
            
            # Step 4: Run model validation and testing
            validation_results, test_results = self.run_model_validation()
            
            # Step 5: Run hyperparameter optimization
            optimizer, opt_results = self.run_hyperparameter_optimization()
            
            # Step 6: Train basic models
            self.train_basic_models()
            
            # Step 7: Train advanced models
            self.train_advanced_models()
            
            # Step 8: Train multi-task model
            self.train_multi_task_model()
            
            # Step 9: Generate predictions
            self.generate_predictions()
            
            # Step 10: Create ensemble
            self.create_ensemble_predictions()
            
            # Step 11: Create submission
            submission_file = self.create_submission_file()
            
            # Step 12: Save comprehensive results
            self.save_comprehensive_results()
            
            total_time = time.time() - start_time
            print(f"\n🎉 Comprehensive Pipeline completed successfully in {total_time:.2f} seconds!")
            print(f"📁 Submission file: {submission_file}")
            print(f"📊 Evaluation results: {self.eval_dir}")
            print(f"🔧 Optimization results: {self.opt_dir}")
            print(f"🧪 Testing results: {self.test_dir}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Comprehensive Pipeline failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function"""
    # Create comprehensive pipeline
    pipeline = ComprehensiveDeepLearningPipeline()
    
    # Run pipeline
    success = pipeline.run_pipeline()
    
    if success:
        print("\n✅ Comprehensive Deep Learning Pipeline completed successfully!")
        print("📂 Check the outputs/ directory for comprehensive results.")
    else:
        print("\n❌ Comprehensive Deep Learning Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
