#!/usr/bin/env python3
"""
Test suite for Dark Neural Networks

This script tests all components of the PyTorch-based dark neural network
implementation to ensure everything works correctly.
"""

import torch
import numpy as np
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dark_neural_network import (
    DarkNeuralNetwork, QuantumDarkLayer, DarkMatterLayer, 
    HiddenVariableLayer, create_sample_dark_network
)


def test_basic_dark_functionality():
    """Test basic functionality of all dark layer types."""
    print("=== Testing Basic Dark Functionality ===\n")
    
    # Test input
    test_input = torch.randn(3, 4)
    print(f"Test input shape: {test_input.shape}")
    
    # Test QuantumDarkLayer
    print("\n1. Testing QuantumDarkLayer:")
    quantum_layer = QuantumDarkLayer(4, 3, hidden_dim=6)
    quantum_layer.eval()
    
    with torch.no_grad():
        quantum_output = quantum_layer(test_input)
    
    print(f"   Input: {test_input.shape} → Output: {quantum_output.shape}")
    print(f"   Output range: [{torch.min(quantum_output):.3f}, {torch.max(quantum_output):.3f}]")
    
    # Test DarkMatterLayer
    print("\n2. Testing DarkMatterLayer:")
    dark_matter_layer = DarkMatterLayer(4, 3, hidden_dim=6)
    dark_matter_layer.eval()
    
    with torch.no_grad():
        dark_matter_output = dark_matter_layer(test_input)
    
    print(f"   Input: {test_input.shape} → Output: {dark_matter_output.shape}")
    print(f"   Output range: [{torch.min(dark_matter_output):.3f}, {torch.max(dark_matter_output):.3f}]")
    
    # Test HiddenVariableLayer
    print("\n3. Testing HiddenVariableLayer:")
    hidden_var_layer = HiddenVariableLayer(4, 3, hidden_dim=6)
    hidden_var_layer.eval()
    
    with torch.no_grad():
        hidden_var_output = hidden_var_layer(test_input)
    
    print(f"   Input: {test_input.shape} → Output: {hidden_var_output.shape}")
    print(f"   Output range: [{torch.min(hidden_var_output):.3f}, {torch.max(hidden_var_output):.3f}]")
    
    return True


def test_dark_network_composition():
    """Test composition of multiple dark layers."""
    print("=== Testing Dark Network Composition ===\n")
    
    test_cases = [
        (1, 5),   # Single sample
        (5, 5),   # Multiple samples
        (10, 5),  # Batch processing
    ]
    
    for i, (batch_size, input_size) in enumerate(test_cases, 1):
        print(f"Test case {i}:")
        
        # Create test network
        network = DarkNeuralNetwork()
        network.add_layer(QuantumDarkLayer(input_size, 8, hidden_dim=10))
        network.add_layer(DarkMatterLayer(8, 6, hidden_dim=8))
        network.add_layer(HiddenVariableLayer(6, 2, hidden_dim=6))
        network.eval()
        
        # Test input
        test_input = torch.randn(batch_size, input_size)
        
        with torch.no_grad():
            output = network(test_input)
        
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output mean: {torch.mean(output):.8f}")
        print(f"   Output std: {torch.std(output):.8f}")
    
    return True


def test_dark_deterministic_behavior():
    """Test that the dark networks are deterministic."""
    print("=== Testing Dark Deterministic Behavior ===\n")
    
    # Create network
    network = create_sample_dark_network()
    network.eval()
    
    # Test input
    test_input = torch.randn(3, 4)
    print(f"Input shape: {test_input.shape}")
    
    # Run multiple times
    with torch.no_grad():
        output1 = network(test_input)
        output2 = network(test_input)
        output3 = network(test_input)
    
    # Check differences
    diff_12 = torch.norm(output1 - output2).item()
    diff_13 = torch.norm(output1 - output3).item()
    diff_23 = torch.norm(output2 - output3).item()
    
    print(f"Output 1 vs 2 difference: {diff_12:.10f}")
    print(f"Output 1 vs 3 difference: {diff_13:.10f}")
    print(f"Output 2 vs 3 difference: {diff_23:.10f}")
    
    is_deterministic = (diff_12 < 1e-10) and (diff_13 < 1e-10) and (diff_23 < 1e-10)
    print(f"Network is deterministic: {is_deterministic}")
    
    return is_deterministic


def test_dark_physics_properties():
    """Test physics-inspired properties of dark operations."""
    print("=== Testing Dark Physics Properties ===\n")
    
    # Test quantum layer properties
    print("1. Quantum Dark Layer Properties:")
    quantum_layer = QuantumDarkLayer(3, 4, hidden_dim=6)
    quantum_layer.eval()
    
    # Test with quantum basis states
    basis_states = torch.tensor([
        [1.0, 0.0, 0.0],  # |0⟩
        [0.0, 1.0, 0.0],  # |1⟩
        [0.0, 0.0, 1.0],  # |2⟩
    ])
    
    with torch.no_grad():
        quantum_outputs = quantum_layer(basis_states)
    
    for i, output in enumerate(quantum_outputs):
        print(f"   Basis state {i}: {output.numpy()}")
    
    # Test dark matter layer properties
    print("\n2. Dark Matter Layer Properties:")
    dark_matter_layer = DarkMatterLayer(2, 3, hidden_dim=4)
    dark_matter_layer.eval()
    
    # Test with different mass scales
    mass_inputs = torch.tensor([
        [1.0, 1.0],    # Equal masses
        [10.0, 1.0],   # Unequal masses
        [0.1, 0.1],    # Small masses
    ])
    
    with torch.no_grad():
        dark_outputs = dark_matter_layer(mass_inputs)
    
    for i, output in enumerate(dark_outputs):
        mass_scale = ["Equal", "Unequal", "Small"][i]
        print(f"   {mass_scale} masses: {output.numpy()}")
    
    # Test hidden variable layer properties
    print("\n3. Hidden Variable Layer Properties:")
    hidden_layer = HiddenVariableLayer(2, 3, hidden_dim=5)
    hidden_layer.eval()
    
    # Test with correlated inputs
    corr_inputs = torch.tensor([
        [1.0, 1.0],    # Positive correlation
        [1.0, -1.0],   # Negative correlation
        [0.0, 0.0],    # Zero point
    ])
    
    with torch.no_grad():
        hidden_outputs = hidden_layer(corr_inputs)
    
    for i, output in enumerate(hidden_outputs):
        corr_type = ["Positive", "Negative", "Zero"][i]
        print(f"   {corr_type} correlation: {output.numpy()}")
    
    return True


def test_dark_edge_cases():
    """Test edge cases and boundary conditions for dark networks."""
    print("=== Testing Dark Edge Cases ===\n")
    
    network = create_sample_dark_network()
    network.eval()
    
    # Test zero input
    print("1. Zero input test:")
    zero_input = torch.zeros(2, 4)
    with torch.no_grad():
        zero_output = network(zero_input)
    print(f"   Input: all zeros, shape {zero_input.shape}")
    print(f"   Output: {zero_output.numpy()}")
    
    # Test very small input
    print("\n2. Small input test:")
    small_input = torch.full((2, 4), 1e-6)
    with torch.no_grad():
        small_output = network(small_input)
    print(f"   Input: 1e-6, shape {small_input.shape}")
    print(f"   Output range: [{torch.min(small_output):.8f}, {torch.max(small_output):.8f}]")
    
    # Test large input
    print("\n3. Large input test:")
    large_input = torch.full((2, 4), 100.0)
    with torch.no_grad():
        large_output = network(large_input)
    print(f"   Input: 100, shape {large_input.shape}")
    print(f"   Output range: [{torch.min(large_output):.6f}, {torch.max(large_output):.6f}]")
    
    # Test single sample
    print("\n4. Single sample test:")
    single_input = torch.randn(4)
    with torch.no_grad():
        single_output = network(single_input)
    print(f"   Input shape: {single_input.shape}")
    print(f"   Output shape: {single_output.shape}")
    
    return True


def test_pytorch_integration():
    """Test PyTorch-specific features and integration."""
    print("=== Testing PyTorch Integration ===\n")
    
    # Test device compatibility
    print("1. Device compatibility:")
    network = create_sample_dark_network()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    # Move to device
    network = network.to(device)
    test_input = torch.randn(3, 4).to(device)
    
    with torch.no_grad():
        output = network(test_input)
    
    print(f"   Input device: {test_input.device}")
    print(f"   Output device: {output.device}")
    print(f"   Output shape: {output.shape}")
    
    # Test gradient computation (even though we don't train)
    print("\n2. Gradient computation:")
    test_input.requires_grad_(True)
    network.train()  # Enable gradient computation
    
    output = network(test_input)
    loss = torch.sum(output)
    loss.backward()
    
    print(f"   Input gradients available: {test_input.grad is not None}")
    print(f"   Gradient norm: {torch.norm(test_input.grad).item():.6f}")
    
    # Test numpy compatibility
    print("\n3. NumPy compatibility:")
    numpy_input = np.random.randn(3, 4)
    network.eval()
    
    with torch.no_grad():
        torch_output = network.predict(numpy_input)
    
    print(f"   NumPy input shape: {numpy_input.shape}")
    print(f"   PyTorch output shape: {torch_output.shape}")
    print(f"   Output type: {type(torch_output)}")
    
    return True


def run_dark_comprehensive_test():
    """Run all dark neural network tests and report results."""
    print("Dark Neural Network Comprehensive Test Suite")
    print("="*60)
    
    tests = [
        ("Basic Dark Functionality", test_basic_dark_functionality),
        ("Dark Network Composition", test_dark_network_composition),
        ("Dark Deterministic Behavior", test_dark_deterministic_behavior),
        ("Dark Physics Properties", test_dark_physics_properties),
        ("Dark Edge Cases", test_dark_edge_cases),
        ("PyTorch Integration", test_pytorch_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name.upper().replace(' ', ' ')}")
        try:
            result = test_func()
            if result:
                print(f"\n✓ {test_name}: PASSED")
                results.append(True)
            else:
                print(f"\n✗ {test_name}: FAILED")
                results.append(False)
        except Exception as e:
            print(f"\n✗ {test_name}: ERROR - {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for i, (test_name, _) in enumerate(tests):
        status = "PASS" if results[i] else "FAIL"
        print(f"{test_name:.<30} {status}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All dark neural network tests passed!")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    success = run_dark_comprehensive_test()
    sys.exit(0 if success else 1)