"""
Deep Learning Hyperparameter Optimization Module
Uses Optuna for automated hyperparameter tuning of deep learning models
"""

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

from .deep_learning_models import LSTMModel, GRUModel, TransformerModel, HybridModel
from .deep_learning_evaluation import DeepLearningEvaluator

class DeepLearningOptimizer:
    """Hyperparameter optimizer for deep learning models"""
    
    def __init__(self, n_trials=100, timeout=3600):
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_params = {}
        self.best_scores = {}
        self.study = None
        
    def create_study(self, study_name, direction='minimize'):
        """Create Optuna study for optimization"""
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner
        )
        
        return self.study
    
    def suggest_lstm_params(self, trial):
        """Suggest hyperparameters for LSTM model"""
        params = {
            'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256, 512]),
            'num_layers': trial.suggest_int('num_layers', 1, 4),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'sequence_length': trial.suggest_int('sequence_length', 10, 30),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        }
        return params
    
    def suggest_gru_params(self, trial):
        """Suggest hyperparameters for GRU model"""
        params = {
            'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256, 512]),
            'num_layers': trial.suggest_int('num_layers', 1, 4),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'sequence_length': trial.suggest_int('sequence_length', 10, 30),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        }
        return params
    
    def suggest_transformer_params(self, trial):
        """Suggest hyperparameters for Transformer model"""
        params = {
            'd_model': trial.suggest_categorical('d_model', [64, 128, 256, 512]),
            'nhead': trial.suggest_categorical('nhead', [4, 8, 16]),
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'dropout': trial.suggest_float('dropout', 0.1, 0.3),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'sequence_length': trial.suggest_int('sequence_length', 10, 30),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        }
        return params
    
    def suggest_hybrid_params(self, trial):
        """Suggest hyperparameters for Hybrid model"""
        params = {
            'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'sequence_length': trial.suggest_int('sequence_length', 10, 30),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        }
        return params
    
    def get_model_params(self, model_type, trial):
        """Get hyperparameters for specific model type"""
        if model_type == 'lstm':
            return self.suggest_lstm_params(trial)
        elif model_type == 'gru':
            return self.suggest_gru_params(trial)
        elif model_type == 'transformer':
            return self.suggest_transformer_params(trial)
        elif model_type == 'hybrid':
            return self.suggest_hybrid_params(trial)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def create_model(self, model_type, input_size, params):
        """Create model with given parameters"""
        if model_type == 'lstm':
            return LSTMModel(
                input_size=input_size,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            )
        elif model_type == 'gru':
            return GRUModel(
                input_size=input_size,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            )
        elif model_type == 'transformer':
            return TransformerModel(
                input_size=input_size,
                d_model=params['d_model'],
                nhead=params['nhead'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            )
        elif model_type == 'hybrid':
            return HybridModel(
                input_size=input_size,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def objective_function(self, trial, model_type, X_train, y_train, X_val, y_val, 
                          input_size, max_epochs=50, patience=10):
        """Objective function for Optuna optimization"""
        try:
            # Get hyperparameters
            params = self.get_model_params(model_type, trial)
            
            # Create model
            model = self.create_model(model_type, input_size, params)
            
            # Create trainer
            from .deep_learning_models import PyTorchTrainer
            trainer = PyTorchTrainer(model)
            
            # Prepare data
            train_loader, scaler = trainer.prepare_data(
                X_train, y_train,
                params['sequence_length'],
                params['batch_size']
            )
            
            # Create validation data
            val_loader, _ = trainer.prepare_data(
                X_val, y_val,
                params['sequence_length'],
                params['batch_size']
            )
            
            # Train model
            train_losses, val_losses = trainer.train(
                train_loader,
                val_loader=val_loader,
                epochs=max_epochs,
                lr=params['learning_rate'],
                patience=patience
            )
            
            # Get best validation loss
            best_val_loss = min(val_losses) if val_losses else float('inf')
            
            # Report intermediate value for pruning
            trial.report(best_val_loss, step=len(val_losses))
            
            return best_val_loss
            
        except Exception as e:
            print(f"Trial failed: {str(e)}")
            return float('inf')
    
    def optimize_model(self, model_type, X_train, y_train, X_val, y_val, 
                      input_size, study_name=None, max_epochs=50, patience=10):
        """Optimize hyperparameters for a specific model type"""
        if study_name is None:
            study_name = f"{model_type}_optimization"
        
        # Create study
        study = self.create_study(study_name, direction='minimize')
        
        # Run optimization
        study.optimize(
            lambda trial: self.objective_function(
                trial, model_type, X_train, y_train, X_val, y_val,
                input_size, max_epochs, patience
            ),
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        # Store results
        self.best_params[model_type] = study.best_params
        self.best_scores[model_type] = study.best_value
        
        print(f"\n{model_type.upper()} Optimization Results:")
        print(f"Best Score: {study.best_value:.6f}")
        print(f"Best Parameters: {study.best_params}")
        
        return study
    
    def optimize_all_models(self, X_train, y_train, X_val, y_val, input_size,
                           model_types=['lstm', 'gru'], max_epochs=50, patience=10):
        """Optimize hyperparameters for all model types"""
        results = {}
        
        for model_type in model_types:
            print(f"\n{'='*50}")
            print(f"Optimizing {model_type.upper()} model...")
            print(f"{'='*50}")
            
            try:
                study = self.optimize_model(
                    model_type, X_train, y_train, X_val, y_val,
                    input_size, max_epochs, max_epochs, patience
                )
                results[model_type] = {
                    'study': study,
                    'best_params': study.best_params,
                    'best_score': study.best_value
                }
            except Exception as e:
                print(f"Error optimizing {model_type}: {str(e)}")
                continue
        
        return results
    
    def get_best_model_config(self, model_type):
        """Get best hyperparameters for a model type"""
        if model_type in self.best_params:
            return self.best_params[model_type]
        else:
            raise ValueError(f"No optimization results for {model_type}")
    
    def create_optimization_report(self, save_path=None):
        """Create comprehensive optimization report"""
        if not self.best_params:
            return "No optimization results available."
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DEEP LEARNING HYPERPARAMETER OPTIMIZATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary of best results
        report_lines.append("OPTIMIZATION SUMMARY:")
        report_lines.append("-" * 40)
        
        for model_type, params in self.best_params.items():
            score = self.best_scores.get(model_type, "N/A")
            report_lines.append(f"{model_type.upper()}:")
            report_lines.append(f"  Best Score: {score}")
            report_lines.append(f"  Best Parameters:")
            for param, value in params.items():
                report_lines.append(f"    {param}: {value}")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS:")
        report_lines.append("-" * 40)
        
        # Find best overall model
        if self.best_scores:
            best_model = min(self.best_scores, key=self.best_scores.get)
            best_score = self.best_scores[best_model]
            report_lines.append(f"Best Overall Model: {best_model.upper()}")
            report_lines.append(f"Best Score: {best_score}")
            report_lines.append("")
        
        # Parameter insights
        report_lines.append("PARAMETER INSIGHTS:")
        report_lines.append("-" * 40)
        
        # Analyze common patterns
        hidden_sizes = [params.get('hidden_size', 0) for params in self.best_params.values()]
        if hidden_sizes:
            most_common_hidden = max(set(hidden_sizes), key=hidden_sizes.count)
            report_lines.append(f"Most Common Hidden Size: {most_common_hidden}")
        
        learning_rates = [params.get('learning_rate', 0) for params in self.best_params.values()]
        if learning_rates:
            avg_lr = np.mean(learning_rates)
            report_lines.append(f"Average Learning Rate: {avg_lr:.6f}")
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def save_optimization_results(self, save_dir):
        """Save optimization results to files"""
        import os
        from pathlib import Path
        import json
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Save best parameters
        params_file = save_dir / "best_hyperparameters.json"
        with open(params_file, 'w') as f:
            json.dump(self.best_params, f, indent=2, default=str)
        
        # Save best scores
        scores_file = save_dir / "best_scores.json"
        with open(scores_file, 'w') as f:
            json.dump(self.best_scores, f, indent=2, default=str)
        
        # Save optimization report
        report_file = save_dir / "optimization_report.txt"
        self.create_optimization_report(str(report_file))
        
        print(f"Optimization results saved to: {save_dir}")
        return save_dir

def optimize_single_model(model_type, X_train, y_train, X_val, y_val, input_size,
                         n_trials=50, max_epochs=50, patience=10):
    """Convenience function to optimize a single model"""
    optimizer = DeepLearningOptimizer(n_trials=n_trials)
    
    study = optimizer.optimize_model(
        model_type, X_train, y_train, X_val, y_val,
        input_size, max_epochs=max_epochs, patience=patience
    )
    
    return optimizer, study

def optimize_multiple_models(X_train, y_train, X_val, y_val, input_size,
                            model_types=['lstm', 'gru'], n_trials=50, max_epochs=50, patience=10):
    """Convenience function to optimize multiple models"""
    optimizer = DeepLearningOptimizer(n_trials=n_trials)
    
    results = optimizer.optimize_all_models(
        X_train, y_train, X_val, y_val, input_size,
        model_types, max_epochs, patience
    )
    
    return optimizer, results
