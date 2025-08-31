"""
Deep Learning Testing and Validation Module
Provides comprehensive testing, validation, and benchmarking capabilities
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import warnings
from pathlib import Path
import json
import unittest
from typing import Dict, List, Any, Tuple
warnings.filterwarnings('ignore')

from .deep_learning_models import LSTMModel, GRUModel, TransformerModel, HybridModel
from .advanced_deep_learning import MultiTaskLSTM, AttentionLSTM, TemporalFusionTransformer
from .deep_learning_evaluation import DeepLearningEvaluator

class DeepLearningTestSuite:
    """Comprehensive test suite for deep learning models"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        
    def test_model_creation(self, model_type, input_size, **kwargs):
        """Test if models can be created without errors"""
        try:
            if model_type == 'lstm':
                model = LSTMModel(input_size, **kwargs)
            elif model_type == 'gru':
                model = GRUModel(input_size, **kwargs)
            elif model_type == 'transformer':
                model = TransformerModel(input_size, **kwargs)
            elif model_type == 'hybrid':
                model = HybridModel(input_size, **kwargs)
            elif model_type == 'multi_task_lstm':
                model = MultiTaskLSTM(input_size, num_targets=5, **kwargs)
            elif model_type == 'attention_lstm':
                model = AttentionLSTM(input_size, **kwargs)
            elif model_type == 'temporal_fusion':
                model = TemporalFusionTransformer(input_size, **kwargs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Test forward pass
            x = torch.randn(2, 10, input_size)  # batch_size=2, seq_len=10, features=input_size
            with torch.no_grad():
                output = model(x)
            
            # Check output shape
            if isinstance(output, dict):
                # Multi-task case
                for key, value in output.items():
                    assert value.shape == (2, 1), f"Output shape mismatch for {key}: {value.shape}"
            else:
                # Single task case
                assert output.shape == (2, 1), f"Output shape mismatch: {output.shape}"
            
            self.test_results[f"{model_type}_creation"] = "PASS"
            return True, f"{model_type} model created and tested successfully"
            
        except Exception as e:
            self.test_results[f"{model_type}_creation"] = "FAIL"
            return False, f"{model_type} model creation failed: {str(e)}"
    
    def test_data_flow(self, model_type, input_size, **kwargs):
        """Test complete data flow through the model"""
        try:
            # Create model
            if model_type == 'lstm':
                model = LSTMModel(input_size, **kwargs)
            elif model_type == 'gru':
                model = GRUModel(input_size, **kwargs)
            elif model_type == 'transformer':
                model = TransformerModel(input_size, **kwargs)
            elif model_type == 'hybrid':
                model = HybridModel(input_size, **kwargs)
            elif model_type == 'multi_task_lstm':
                model = MultiTaskLSTM(input_size, num_targets=5, **kwargs)
            elif model_type == 'attention_lstm':
                model = AttentionLSTM(input_size, **kwargs)
            elif model_type == 'temporal_fusion':
                model = TemporalFusionTransformer(input_size, **kwargs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Create test data
            batch_size = 4
            seq_len = 15
            x = torch.randn(batch_size, seq_len, input_size)
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                output = model(x)
            
            # Test backward pass (gradient computation)
            model.train()
            if isinstance(output, dict):
                # Multi-task case
                loss = sum(torch.mean(out**2) for out in output.values())
            else:
                # Single task case
                loss = torch.mean(output**2)
            
            loss.backward()
            
            # Check if gradients were computed
            has_gradients = any(p.grad is not None for p in model.parameters())
            assert has_gradients, "No gradients computed during backward pass"
            
            self.test_results[f"{model_type}_data_flow"] = "PASS"
            return True, f"{model_type} data flow test passed"
            
        except Exception as e:
            self.test_results[f"{model_type}_data_flow"] = "FAIL"
            return False, f"{model_type} data flow test failed: {str(e)}"
    
    def test_model_saving_loading(self, model_type, input_size, **kwargs):
        """Test model saving and loading functionality"""
        try:
            # Create model
            if model_type == 'lstm':
                model = LSTMModel(input_size, **kwargs)
            elif model_type == 'gru':
                model = GRUModel(input_size, **kwargs)
            elif model_type == 'transformer':
                model = TransformerModel(input_size, **kwargs)
            elif model_type == 'hybrid':
                model = HybridModel(input_size, **kwargs)
            elif model_type == 'multi_task_lstm':
                model = MultiTaskLSTM(input_size, num_targets=5, **kwargs)
            elif model_type == 'attention_lstm':
                model = AttentionLSTM(input_size, **kwargs)
            elif model_type == 'temporal_fusion':
                model = TemporalFusionTransformer(input_size, **kwargs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Create test data
            x = torch.randn(2, 10, input_size)
            
            # Get original output
            model.eval()
            with torch.no_grad():
                original_output = model(x)
            
            # Save model
            temp_path = Path("temp_model.pth")
            torch.save(model.state_dict(), temp_path)
            
            # Load model
            loaded_model = type(model)(input_size, **kwargs)
            loaded_model.load_state_dict(torch.load(temp_path))
            
            # Test loaded model
            loaded_model.eval()
            with torch.no_grad():
                loaded_output = loaded_model(x)
            
            # Compare outputs
            if isinstance(original_output, dict):
                # Multi-task case
                for key in original_output.keys():
                    assert torch.allclose(original_output[key], loaded_output[key], atol=1e-6), \
                        f"Output mismatch for {key} after loading"
            else:
                # Single task case
                assert torch.allclose(original_output, loaded_output, atol=1e-6), \
                    "Output mismatch after loading"
            
            # Clean up
            temp_path.unlink()
            
            self.test_results[f"{model_type}_save_load"] = "PASS"
            return True, f"{model_type} save/load test passed"
            
        except Exception as e:
            self.test_results[f"{model_type}_save_load"] = "FAIL"
            return False, f"{model_type} save/load test failed: {str(e)}"
    
    def benchmark_model_performance(self, model_type, input_size, batch_sizes=[1, 4, 16, 32], **kwargs):
        """Benchmark model performance across different batch sizes"""
        try:
            # Create model
            if model_type == 'lstm':
                model = LSTMModel(input_size, **kwargs)
            elif model_type == 'gru':
                model = GRUModel(input_size, **kwargs)
            elif model_type == 'transformer':
                model = TransformerModel(input_size, **kwargs)
            elif model_type == 'hybrid':
                model = HybridModel(input_size, **kwargs)
            elif model_type == 'multi_task_lstm':
                model = MultiTaskLSTM(input_size, num_targets=5, **kwargs)
            elif model_type == 'attention_lstm':
                model = AttentionLSTM(input_size, **kwargs)
            elif model_type == 'temporal_fusion':
                model = TemporalFusionTransformer(input_size, **kwargs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model.eval()
            device = next(model.parameters()).device
            
            benchmark_results = {}
            
            for batch_size in batch_sizes:
                # Create test data
                x = torch.randn(batch_size, 20, input_size).to(device)
                
                # Warmup
                with torch.no_grad():
                    for _ in range(5):
                        _ = model(x)
                
                # Benchmark forward pass
                torch.cuda.synchronize() if device.type == 'cuda' else None
                start_time = time.time()
                
                with torch.no_grad():
                    for _ in range(100):
                        _ = model(x)
                
                torch.cuda.synchronize() if device.type == 'cuda' else None
                end_time = time.time()
                
                # Calculate metrics
                total_time = end_time - start_time
                avg_time = total_time / 100
                throughput = batch_size / avg_time
                
                benchmark_results[batch_size] = {
                    'avg_inference_time': avg_time * 1000,  # Convert to milliseconds
                    'throughput': throughput,
                    'total_time': total_time
                }
            
            self.performance_metrics[model_type] = benchmark_results
            self.test_results[f"{model_type}_benchmark"] = "PASS"
            
            return True, benchmark_results
            
        except Exception as e:
            self.test_results[f"{model_type}_benchmark"] = "FAIL"
            return False, f"{model_type} benchmark failed: {str(e)}"
    
    def run_comprehensive_tests(self, input_size=100):
        """Run all tests for all model types"""
        print("Running comprehensive deep learning model tests...")
        print("=" * 60)
        
        model_types = [
            'lstm', 'gru', 'transformer', 'hybrid',
            'multi_task_lstm', 'attention_lstm', 'temporal_fusion'
        ]
        
        test_functions = [
            self.test_model_creation,
            self.test_data_flow,
            self.test_model_saving_loading,
            self.benchmark_model_performance
        ]
        
        for model_type in model_types:
            print(f"\nTesting {model_type.upper()} model...")
            print("-" * 40)
            
            for test_func in test_functions:
                if test_func == self.benchmark_model_performance:
                    success, result = test_func(model_type, input_size)
                else:
                    success, result = test_func(model_type, input_size)
                
                if success:
                    print(f"✓ {test_func.__name__}: PASS")
                    if test_func == self.benchmark_model_performance:
                        print(f"  Performance results: {len(result)} batch sizes tested")
                else:
                    print(f"✗ {test_func.__name__}: FAIL")
                    print(f"  Error: {result}")
        
        # Print summary
        self.print_test_summary()
        
        return self.test_results
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result == "PASS")
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\nFailed Tests:")
            for test_name, result in self.test_results.items():
                if result == "FAIL":
                    print(f"  - {test_name}")
        
        # Performance summary
        if self.performance_metrics:
            print("\nPerformance Summary:")
            for model_type, metrics in self.performance_metrics.items():
                print(f"\n{model_type.upper()}:")
                for batch_size, perf in metrics.items():
                    print(f"  Batch {batch_size}: {perf['avg_inference_time']:.2f}ms, "
                          f"{perf['throughput']:.1f} samples/sec")
    
    def save_test_results(self, save_dir):
        """Save test results to files"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Save test results
        test_file = save_dir / "test_results.json"
        with open(test_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        # Save performance metrics
        perf_file = save_dir / "performance_metrics.json"
        with open(perf_file, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2, default=str)
        
        print(f"Test results saved to: {save_dir}")
        return save_dir

class ModelValidator:
    """Model validation and sanity checking"""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_model_architecture(self, model, input_size, expected_output_size=1):
        """Validate model architecture and dimensions"""
        try:
            # Test with different input sizes
            test_sizes = [1, 4, 16]
            results = {}
            
            for batch_size in test_sizes:
                x = torch.randn(batch_size, 10, input_size)
                with torch.no_grad():
                    output = model(x)
                
                if isinstance(output, dict):
                    # Multi-task case
                    for key, value in output.items():
                        expected_shape = (batch_size, expected_output_size)
                        assert value.shape == expected_shape, \
                            f"Shape mismatch for {key}: expected {expected_shape}, got {value.shape}"
                else:
                    # Single task case
                    expected_shape = (batch_size, expected_output_size)
                    assert output.shape == expected_shape, \
                        f"Shape mismatch: expected {expected_shape}, got {output.shape}"
                
                results[batch_size] = "PASS"
            
            return True, results
            
        except Exception as e:
            return False, str(e)
    
    def validate_gradient_flow(self, model, input_size):
        """Validate that gradients flow properly through the model"""
        try:
            model.train()
            x = torch.randn(4, 10, input_size)
            
            # Forward pass
            output = model(x)
            
            # Create loss
            if isinstance(output, dict):
                loss = sum(torch.mean(out**2) for out in output.values())
            else:
                loss = torch.mean(output**2)
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            has_gradients = any(p.grad is not None for p in model.parameters())
            gradient_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            
            if not has_gradients:
                return False, "No gradients computed"
            
            if all(norm == 0 for norm in gradient_norms):
                return False, "All gradients are zero"
            
            return True, {
                'has_gradients': has_gradients,
                'num_parameters_with_gradients': len(gradient_norms),
                'avg_gradient_norm': np.mean(gradient_norms)
            }
            
        except Exception as e:
            return False, str(e)
    
    def validate_model_consistency(self, model, input_size, num_runs=5):
        """Validate that model produces consistent outputs for same input"""
        try:
            model.eval()
            x = torch.randn(4, 10, input_size)
            
            outputs = []
            with torch.no_grad():
                for _ in range(num_runs):
                    output = model(x)
                    outputs.append(output)
            
            # Check consistency
            if isinstance(outputs[0], dict):
                # Multi-task case
                consistency_results = {}
                for key in outputs[0].keys():
                    key_outputs = [out[key] for out in outputs]
                    max_diff = max(torch.max(torch.abs(key_outputs[i] - key_outputs[i-1])).item() 
                                 for i in range(1, len(key_outputs)))
                    consistency_results[key] = max_diff
                
                is_consistent = all(diff < 1e-6 for diff in consistency_results.values())
            else:
                # Single task case
                max_diff = max(torch.max(torch.abs(outputs[i] - outputs[i-1])).item() 
                             for i in range(1, len(outputs)))
                consistency_results = {'single_output': max_diff}
                is_consistent = max_diff < 1e-6
            
            return is_consistent, consistency_results
            
        except Exception as e:
            return False, str(e)

def run_quick_validation():
    """Run quick validation tests for all models"""
    print("Running quick validation tests...")
    
    validator = ModelValidator()
    input_size = 50
    
    model_configs = {
        'lstm': (LSTMModel, {'hidden_size': 64, 'num_layers': 2}),
        'gru': (GRUModel, {'hidden_size': 64, 'num_layers': 2}),
        'transformer': (TransformerModel, {'d_model': 64, 'nhead': 4, 'num_layers': 2}),
        'hybrid': (HybridModel, {'hidden_size': 64, 'num_layers': 2}),
        'multi_task_lstm': (MultiTaskLSTM, {'hidden_size': 64, 'num_layers': 2, 'num_targets': 3}),
        'attention_lstm': (AttentionLSTM, {'hidden_size': 64, 'num_layers': 2}),
        'temporal_fusion': (TemporalFusionTransformer, {'d_model': 64, 'nhead': 4, 'num_layers': 2})
    }
    
    results = {}
    
    for model_name, (model_class, config) in model_configs.items():
        print(f"\nValidating {model_name}...")
        
        try:
            model = model_class(input_size, **config)
            
            # Architecture validation
            arch_valid, arch_result = validator.validate_model_architecture(model, input_size)
            
            # Gradient flow validation
            grad_valid, grad_result = validator.validate_gradient_flow(model, input_size)
            
            # Consistency validation
            cons_valid, cons_result = validator.validate_model_consistency(model, input_size)
            
            results[model_name] = {
                'architecture': arch_valid,
                'gradient_flow': grad_valid,
                'consistency': cons_valid,
                'details': {
                    'architecture': arch_result,
                    'gradient_flow': grad_result,
                    'consistency': cons_result
                }
            }
            
            status = "PASS" if all([arch_valid, grad_valid, cons_valid]) else "FAIL"
            print(f"  Status: {status}")
            
        except Exception as e:
            results[model_name] = {
                'error': str(e),
                'status': 'ERROR'
            }
            print(f"  Status: ERROR - {str(e)}")
    
    return results

if __name__ == "__main__":
    # Run quick validation
    validation_results = run_quick_validation()
    
    # Run comprehensive tests
    test_suite = DeepLearningTestSuite()
    test_results = test_suite.run_comprehensive_tests()
    
    print("\nValidation and testing completed!")
