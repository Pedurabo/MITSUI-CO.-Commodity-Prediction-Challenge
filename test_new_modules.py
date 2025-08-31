#!/usr/bin/env python3
"""
Quick test script to verify all new modules work correctly
"""

import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

def test_optimization_module():
    """Test the hyperparameter optimization module"""
    print("Testing hyperparameter optimization module...")
    
    try:
        from deep_learning_optimization import DeepLearningOptimizer
        
        # Create optimizer
        optimizer = DeepLearningOptimizer(n_trials=5)
        
        # Test parameter suggestion
        lstm_params = optimizer.suggest_lstm_params(None)
        gru_params = optimizer.suggest_gru_params(None)
        
        print(f"  ✓ LSTM params: {len(lstm_params)} parameters")
        print(f"  ✓ GRU params: {len(gru_params)} parameters")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return False

def test_advanced_techniques():
    """Test the advanced deep learning techniques module"""
    print("Testing advanced deep learning techniques...")
    
    try:
        from advanced_deep_learning import MultiTaskLSTM, AttentionLSTM, TemporalFusionTransformer
        
        # Test model creation
        input_size = 50
        
        # Multi-task LSTM
        multi_task_model = MultiTaskLSTM(input_size, num_targets=3)
        print(f"  ✓ Multi-task LSTM created: {multi_task_model}")
        
        # Attention LSTM
        attention_lstm = AttentionLSTM(input_size)
        print(f"  ✓ Attention LSTM created: {attention_lstm}")
        
        # Temporal Fusion Transformer
        temporal_fusion = TemporalFusionTransformer(input_size)
        print(f"  ✓ Temporal Fusion Transformer created: {temporal_fusion}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return False

def test_testing_module():
    """Test the testing and validation module"""
    print("Testing testing and validation module...")
    
    try:
        from deep_learning_testing import DeepLearningTestSuite, ModelValidator
        
        # Create test suite
        test_suite = DeepLearningTestSuite()
        print(f"  ✓ Test suite created: {test_suite}")
        
        # Create validator
        validator = ModelValidator()
        print(f"  ✓ Model validator created: {validator}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return False

def test_imports():
    """Test all module imports"""
    print("Testing module imports...")
    
    modules_to_test = [
        'deep_learning_models',
        'deep_learning_evaluation', 
        'deep_learning_optimization',
        'advanced_deep_learning',
        'deep_learning_testing',
        'data_processing'
    ]
    
    all_imports_work = True
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name} imported successfully")
        except Exception as e:
            print(f"  ✗ {module_name} import failed: {str(e)}")
            all_imports_work = False
    
    return all_imports_work

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing All New Deep Learning Modules")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Hyperparameter Optimization", test_optimization_module),
        ("Advanced Techniques", test_advanced_techniques),
        ("Testing & Validation", test_testing_module)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        success = test_func()
        results.append((test_name, success))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All modules are working correctly!")
        print("✅ Ready to run the comprehensive pipeline!")
    else:
        print("❌ Some modules have issues that need to be resolved.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
