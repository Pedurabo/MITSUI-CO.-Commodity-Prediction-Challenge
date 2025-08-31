"""
Deep Learning Model Evaluation Module
Provides comprehensive evaluation metrics and visualization tools for time series models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

class DeepLearningEvaluator:
    """Comprehensive evaluator for deep learning time series models"""
    
    def __init__(self):
        self.metrics = {}
        self.predictions = {}
        self.actuals = {}
        
    def calculate_basic_metrics(self, y_true, y_pred, target_name):
        """Calculate basic regression metrics"""
        metrics = {}
        
        # Basic regression metrics
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Correlation metrics
        pearson_corr, pearson_p = pearsonr(y_true, y_pred)
        spearman_corr, spearman_p = spearmanr(y_true, y_pred)
        
        metrics['pearson_corr'] = pearson_corr
        metrics['pearson_p'] = pearson_p
        metrics['spearman_corr'] = spearman_corr
        metrics['spearman_p'] = spearman_p
        
        # Time series specific metrics
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        metrics['smape'] = 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
        
        # Direction accuracy (for financial time series)
        direction_actual = np.diff(y_true) > 0
        direction_pred = np.diff(y_pred) > 0
        metrics['direction_accuracy'] = np.mean(direction_actual == direction_pred)
        
        # Volatility accuracy
        vol_actual = np.std(y_true)
        vol_pred = np.std(y_pred)
        metrics['volatility_ratio'] = vol_pred / (vol_actual + 1e-8)
        
        self.metrics[target_name] = metrics
        return metrics
    
    def calculate_competition_metric(self, y_true, y_pred, target_name):
        """Calculate competition-specific Sharpe ratio variant"""
        # Spearman correlation
        spearman_corr, _ = spearmanr(y_true, y_pred)
        
        # Calculate stability metric (lower variance is better)
        # For competition: mean_spearman / std_spearman
        # Here we'll use a simplified version
        
        # Calculate rolling correlations for stability
        window_size = min(20, len(y_true) // 4)
        if window_size > 5:
            rolling_corrs = []
            for i in range(window_size, len(y_true)):
                window_true = y_true[i-window_size:i]
                window_pred = y_pred[i-window_size:i]
                if len(window_true) > 1:
                    corr, _ = spearmanr(window_true, window_pred)
                    if not np.isnan(corr):
                        rolling_corrs.append(corr)
            
            if rolling_corrs:
                stability_metric = np.mean(rolling_corrs) / (np.std(rolling_corrs) + 1e-8)
            else:
                stability_metric = spearman_corr
        else:
            stability_metric = spearman_corr
        
        competition_metric = {
            'spearman_correlation': spearman_corr,
            'stability_metric': stability_metric,
            'competition_score': spearman_corr / (np.std(y_pred) + 1e-8)
        }
        
        if target_name not in self.metrics:
            self.metrics[target_name] = {}
        self.metrics[target_name].update(competition_metric)
        
        return competition_metric
    
    def plot_predictions_vs_actual(self, y_true, y_pred, target_name, model_name, save_path=None):
        """Plot predictions vs actual values"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scatter plot
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=20)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title(f'{target_name} - {model_name}\nPredictions vs Actual')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Time series plot
        axes[0, 1].plot(y_true, label='Actual', alpha=0.8)
        axes[0, 1].plot(y_pred, label='Predicted', alpha=0.8)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Values')
        axes[0, 1].set_title(f'{target_name} - {model_name}\nTime Series Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_pred - y_true
        axes[1, 0].scatter(y_true, residuals, alpha=0.6, s=20)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[1, 0].set_xlabel('Actual Values')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title(f'{target_name} - {model_name}\nResiduals Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.8)
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title(f'{target_name} - {model_name}\nResiduals Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    def plot_training_curves(self, train_losses, val_losses, target_name, model_name, save_path=None):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(12, 5))
        
        epochs = range(len(train_losses))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', alpha=0.8)
        if val_losses:
            plt.plot(epochs, val_losses, 'r-', label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{target_name} - {model_name}\nTraining Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if val_losses:
            plt.subplot(1, 2, 2)
            plt.plot(epochs, np.array(train_losses) - np.array(val_losses), 'g-', alpha=0.8)
            plt.xlabel('Epoch')
            plt.ylabel('Train Loss - Val Loss')
            plt.title(f'{target_name} - {model_name}\nLoss Difference')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return plt.gcf()
    
    def create_model_comparison_table(self, target_name):
        """Create a comparison table for different models on the same target"""
        if target_name not in self.metrics:
            return None
        
        metrics_df = pd.DataFrame(self.metrics[target_name]).T
        
        # Round numeric values
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        metrics_df[numeric_cols] = metrics_df[numeric_cols].round(6)
        
        return metrics_df
    
    def create_summary_report(self, save_path=None):
        """Create a comprehensive summary report"""
        if not self.metrics:
            return "No metrics available for summary report."
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DEEP LEARNING MODEL EVALUATION SUMMARY REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Overall statistics
        all_spearman = []
        all_rmse = []
        all_r2 = []
        
        for target_name, target_metrics in self.metrics.items():
            if isinstance(target_metrics, dict):
                if 'spearman_correlation' in target_metrics:
                    all_spearman.append(target_metrics['spearman_correlation'])
                if 'rmse' in target_metrics:
                    all_rmse.append(target_metrics['rmse'])
                if 'r2' in target_metrics:
                    all_r2.append(target_metrics['r2'])
        
        if all_spearman:
            report_lines.append(f"OVERALL PERFORMANCE SUMMARY:")
            report_lines.append(f"  Average Spearman Correlation: {np.mean(all_spearman):.6f} ± {np.std(all_spearman):.6f}")
            report_lines.append(f"  Average RMSE: {np.mean(all_rmse):.6f} ± {np.std(all_rmse):.6f}")
            report_lines.append(f"  Average R²: {np.mean(all_r2):.6f} ± {np.std(all_r2):.6f}")
            report_lines.append("")
        
        # Per-target breakdown
        for target_name, target_metrics in self.metrics.items():
            if isinstance(target_metrics, dict):
                report_lines.append(f"TARGET: {target_name}")
                report_lines.append("-" * 40)
                
                # Key metrics
                key_metrics = ['rmse', 'mae', 'r2', 'spearman_correlation', 'direction_accuracy']
                for metric in key_metrics:
                    if metric in target_metrics:
                        report_lines.append(f"  {metric.upper()}: {target_metrics[metric]:.6f}")
                
                report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def save_evaluation_results(self, save_dir):
        """Save all evaluation results to files"""
        import os
        from pathlib import Path
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Save metrics as JSON
        import json
        metrics_file = save_dir / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        # Save summary report
        report_file = save_dir / "evaluation_summary.txt"
        self.create_summary_report(str(report_file))
        
        # Save comparison tables
        for target_name in self.metrics:
            if isinstance(self.metrics[target_name], dict):
                table = self.create_model_comparison_table(target_name)
                if table is not None:
                    table_file = save_dir / f"comparison_table_{target_name}.csv"
                    table.to_csv(table_file)
        
        print(f"Evaluation results saved to: {save_dir}")
        return save_dir

def evaluate_model_performance(y_true, y_pred, target_name, model_name, evaluator=None):
    """Convenience function to evaluate a single model"""
    if evaluator is None:
        evaluator = DeepLearningEvaluator()
    
    # Calculate metrics
    basic_metrics = evaluator.calculate_basic_metrics(y_true, y_pred, f"{target_name}_{model_name}")
    competition_metrics = evaluator.calculate_competition_metric(y_true, y_pred, f"{target_name}_{model_name}")
    
    # Print summary
    print(f"\n{target_name} - {model_name} Performance:")
    print(f"  RMSE: {basic_metrics['rmse']:.6f}")
    print(f"  MAE: {basic_metrics['mae']:.6f}")
    print(f"  R²: {basic_metrics['r2']:.6f}")
    print(f"  Spearman Correlation: {basic_metrics['spearman_corr']:.6f}")
    print(f"  Direction Accuracy: {basic_metrics['direction_accuracy']:.6f}")
    print(f"  Competition Score: {competition_metrics['competition_score']:.6f}")
    
    return evaluator, basic_metrics, competition_metrics
