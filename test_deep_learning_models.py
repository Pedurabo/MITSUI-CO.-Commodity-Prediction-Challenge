#!/usr/bin/env python3
"""
Test script for deep learning models
This script tests the basic functionality of our deep learning models
"""

import sys
sys.path.append('src')

import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

def test_pytorch_models():
    """Test PyTorch models"""
    print("Testing PyTorch Models...")
    
    try:
        from deep_learning_models import (
            LSTMModel, GRUModel, TransformerModel, HybridModel
        )
        
        # Test parameters
        input_size = 100
        batch_size = 4
        sequence_length = 10
        
        # Test LSTM
        print("  Testing LSTM...")
        lstm = LSTMModel(input_size, hidden_size=64, num_layers=2)
        x = torch.randn(batch_size, sequence_length, input_size)
        output = lstm(x)
        print(f"    LSTM input shape: {x.shape}, output shape: {output.shape}")
        
        # Test GRU
        print("  Testing GRU...")
        gru = GRUModel(input_size, hidden_size=64, num_layers=2)
        output = gru(x)
        print(f"    GRU input shape: {x.shape}, output shape: {output.shape}")
        
        # Test Transformer
        print("  Testing Transformer...")
        transformer = TransformerModel(input_size, d_model=64, nhead=4, num_layers=2)
        output = transformer(x)
        print(f"    Transformer input shape: {x.shape}, output shape: {output.shape}")
        
        # Test Hybrid
        print("  Testing Hybrid...")
        hybrid = HybridModel(input_size, hidden_size=64, num_layers=2)
        output = hybrid(x)
        print(f"    Hybrid input shape: {x.shape}, output shape: {output.shape}")
        
        print("  ✓ All PyTorch models working correctly!")
        return True
        
    except Exception as e:
        print(f"  ✗ Error testing PyTorch models: {str(e)}")
        return False

def test_tensorflow_models():
    """Test TensorFlow/Keras models"""
    print("Testing TensorFlow/Keras Models...")
    
    try:
        import tensorflow as tf
        from deep_learning_models import (
            create_lstm_keras, create_gru_keras, create_transformer_keras
        )
        
        # Test parameters
        input_shape = (10, 100)  # (sequence_length, features)
        
        # Test LSTM
        print("  Testing LSTM Keras...")
        lstm_keras = create_lstm_keras(input_shape, units=64, layers=2)
        x = tf.random.normal((4, 10, 100))  # (batch, seq, features)
        output = lstm_keras(x)
        print(f"    LSTM Keras input shape: {x.shape}, output shape: {output.shape}")
        
        # Test GRU
        print("  Testing GRU Keras...")
        gru_keras = create_gru_keras(input_shape, units=64, layers=2)
        output = gru_keras(x)
        print(f"    GRU Keras input shape: {x.shape}, output shape: {output.shape}")
        
        # Test Transformer
        print("  Testing Transformer Keras...")
        transformer_keras = create_transformer_keras(input_shape, d_model=64, num_heads=4, num_layers=2)
        output = transformer_keras(x)
        print(f"    Transformer Keras input shape: {x.shape}, output shape: {output.shape}")
        
        print("  ✓ All TensorFlow/Keras models working correctly!")
        return True
        
    except Exception as e:
        print(f"  ✗ Error testing TensorFlow models: {str(e)}")
        return False

def test_trainers():
    """Test trainer classes"""
    print("Testing Trainer Classes...")
    
    try:
        from deep_learning_models import PyTorchTrainer, KerasTrainer
        
        # Test PyTorch trainer
        print("  Testing PyTorch Trainer...")
        from deep_learning_models import LSTMModel
        model = LSTMModel(100, hidden_size=64)
        trainer = PyTorchTrainer(model)
        print(f"    PyTorch trainer created successfully")
        
        # Test Keras trainer
        print("  Testing Keras Trainer...")
        import tensorflow as tf
        from deep_learning_models import create_lstm_keras
        keras_model = create_lstm_keras((10, 100), units=64)
        keras_trainer = KerasTrainer(keras_model)
        print(f"    Keras trainer created successfully")
        
        print("  ✓ All trainers working correctly!")
        return True
        
    except Exception as e:
        print(f"  ✗ Error testing trainers: {str(e)}")
        return False

def test_device_availability():
    """Test device availability"""
    print("Testing Device Availability...")
    
    # PyTorch
    if torch.cuda.is_available():
        print(f"  PyTorch: CUDA available - {torch.cuda.get_device_name(0)}")
        print(f"    CUDA version: {torch.version.cuda}")
        print(f"    GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  PyTorch: CUDA not available, using CPU")
    
    # TensorFlow
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  TensorFlow: {len(gpus)} GPU(s) available")
            for gpu in gpus:
                print(f"    {gpu.name}")
        else:
            print("  TensorFlow: No GPUs available, using CPU")
    except:
        print("  TensorFlow: Could not check GPU availability")

def main():
    """Main test function"""
    print("=" * 60)
    print("Deep Learning Models Test Suite")
    print("=" * 60)
    
    # Test device availability
    test_device_availability()
    print()
    
    # Test PyTorch models
    pytorch_success = test_pytorch_models()
    print()
    
    # Test TensorFlow models
    tensorflow_success = test_tensorflow_models()
    print()
    
    # Test trainers
    trainer_success = test_trainers()
    print()
    
    # Summary
    print("=" * 60)
    print("Test Summary:")
    print(f"  PyTorch Models: {'✓ PASS' if pytorch_success else '✗ FAIL'}")
    print(f"  TensorFlow Models: {'✓ PASS' if tensorflow_success else '✗ FAIL'}")
    print(f"  Trainers: {'✓ PASS' if trainer_success else '✗ FAIL'}")
    
    if all([pytorch_success, tensorflow_success, trainer_success]):
        print("\n🎉 All tests passed! Deep learning models are ready to use.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
