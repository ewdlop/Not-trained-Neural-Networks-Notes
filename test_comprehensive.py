#!/usr/bin/env python3
"""
Comprehensive test suite for Algebraic Neural Networks

This script tests all components of the algebraic neural network implementation
to ensure everything works correctly together.
"""

import sys
import os
import numpy as np

# Add the parent directory to the path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all our implementations
from algebraic_neural_network import (
    AlgebraicNeuralNetwork, PolynomialLayer, GroupTheoryLayer, 
    GeometricAlgebraLayer, create_sample_network, TORCH_AVAILABLE
)

# Try to import AnyonicLayer if PyTorch is available
if TORCH_AVAILABLE:
    from algebraic_neural_network import AnyonicLayer

def test_basic_functionality():
    """Test basic functionality of all layer types."""
    print("=== Testing Basic Functionality ===\n")
    
    # Test data
    test_input = np.random.randn(3, 4)
    print(f"Test input shape: {test_input.shape}")
    
    # Test individual layers
    print("\n1. Testing PolynomialLayer:")
    poly_layer = PolynomialLayer(4, 3, degree=2)
    poly_output = poly_layer.forward(test_input)
    print(f"   Input: {test_input.shape} → Output: {poly_output.shape}")
    print(f"   Output range: [{np.min(poly_output):.3f}, {np.max(poly_output):.3f}]")
    
    print("\n2. Testing GroupTheoryLayer:")
    group_layer = GroupTheoryLayer(4, 3, group_order=6)
    group_output = group_layer.forward(test_input)
    print(f"   Input: {test_input.shape} → Output: {group_output.shape}")
    print(f"   Output range: [{np.min(group_output):.3f}, {np.max(group_output):.3f}]")
    
    print("\n3. Testing GeometricAlgebraLayer:")
    geo_layer = GeometricAlgebraLayer(4, 3)
    geo_output = geo_layer.forward(test_input)
    print(f"   Input: {test_input.shape} → Output: {geo_output.shape}")
    print(f"   Output range: [{np.min(geo_output):.3f}, {np.max(geo_output):.3f}]")
    
    # Test AnyonicLayer if PyTorch is available
    if TORCH_AVAILABLE:
        print("\n4. Testing AnyonicLayer:")
        anyonic_layer = AnyonicLayer(4, 3, anyon_type="fibonacci")
        anyonic_output = anyonic_layer.forward(test_input)
        print(f"   Input: {test_input.shape} → Output: {anyonic_output.shape}")
        print(f"   Output range: [{np.min(anyonic_output):.3f}, {np.max(anyonic_output):.3f}]")
    else:
        print("\n4. AnyonicLayer: Skipping (PyTorch not available)")
    
    return True

def test_network_composition():
    """Test composition of multiple algebraic layers."""
    print("\n=== Testing Network Composition ===\n")
    
    # Create a complete network
    network = AlgebraicNeuralNetwork()
    network.add_layer(PolynomialLayer(5, 8, degree=2))
    network.add_layer(GroupTheoryLayer(8, 6, group_order=8))
    network.add_layer(GeometricAlgebraLayer(6, 3))
    network.add_layer(PolynomialLayer(3, 2, degree=1))
    
    # Test with different input sizes
    test_cases = [
        np.random.randn(1, 5),   # Single sample
        np.random.randn(5, 5),   # Multiple samples
        np.random.randn(10, 5),  # Larger batch
    ]
    
    for i, test_case in enumerate(test_cases):
        output = network.predict(test_case)
        print(f"Test case {i+1}:")
        print(f"   Input shape: {test_case.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output mean: {np.mean(output):.4f}")
        print(f"   Output std: {np.std(output):.4f}")
    
    return True

def test_deterministic_behavior():
    """Test that the networks are deterministic."""
    print("\n=== Testing Deterministic Behavior ===\n")
    
    # Create network
    network = create_sample_network()
    
    # Same input should produce same output
    test_input = np.random.randn(3, 4)
    
    output1 = network.predict(test_input)
    output2 = network.predict(test_input)
    output3 = network.predict(test_input)
    
    # Check if outputs are identical
    diff_12 = np.linalg.norm(output1 - output2)
    diff_13 = np.linalg.norm(output1 - output3)
    diff_23 = np.linalg.norm(output2 - output3)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output 1 vs 2 difference: {diff_12:.10f}")
    print(f"Output 1 vs 3 difference: {diff_13:.10f}")
    print(f"Output 2 vs 3 difference: {diff_23:.10f}")
    
    is_deterministic = (diff_12 < 1e-10) and (diff_13 < 1e-10) and (diff_23 < 1e-10)
    print(f"Network is deterministic: {is_deterministic}")
    
    return is_deterministic

def test_mathematical_properties():
    """Test mathematical properties of algebraic operations."""
    print("\n=== Testing Mathematical Properties ===\n")
    
    # Test polynomial layer properties
    print("1. Polynomial Layer Properties:")
    poly_layer = PolynomialLayer(2, 3, degree=2)
    
    # Linearity test (for degree 1 components)
    x1 = np.array([[1, 0]])
    x2 = np.array([[0, 1]])
    x_sum = np.array([[1, 1]])
    
    y1 = poly_layer.forward(x1)
    y2 = poly_layer.forward(x2)
    y_sum = poly_layer.forward(x_sum)
    
    # Note: Due to higher degree terms, this won't be exactly linear
    linearity_error = np.linalg.norm(y_sum - (y1 + y2))
    print(f"   Linearity deviation (expected for degree > 1): {linearity_error:.4f}")
    
    # Test group theory layer properties
    print("\n2. Group Theory Layer Properties:")
    group_layer = GroupTheoryLayer(2, 4, group_order=4)
    
    # Test with unit vectors
    unit_x = np.array([[1, 0]])
    unit_y = np.array([[0, 1]])
    
    out_x = group_layer.forward(unit_x)
    out_y = group_layer.forward(unit_y)
    
    print(f"   Unit X output norm: {np.linalg.norm(out_x):.4f}")
    print(f"   Unit Y output norm: {np.linalg.norm(out_y):.4f}")
    
    # Test geometric algebra layer properties
    print("\n3. Geometric Algebra Layer Properties:")
    geo_layer = GeometricAlgebraLayer(3, 4)
    
    # Test with orthogonal vectors
    e1 = np.array([[1, 0, 0]])
    e2 = np.array([[0, 1, 0]])
    e3 = np.array([[0, 0, 1]])
    
    out1 = geo_layer.forward(e1)
    out2 = geo_layer.forward(e2)
    out3 = geo_layer.forward(e3)
    
    print(f"   e1 output: {out1[0]}")
    print(f"   e2 output: {out2[0]}")
    print(f"   e3 output: {out3[0]}")
    
    return True

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n=== Testing Edge Cases ===\n")
    
    network = create_sample_network()
    
    # Test with zero input
    zero_input = np.zeros((2, 4))
    zero_output = network.predict(zero_input)
    print(f"1. Zero input test:")
    print(f"   Input: all zeros, shape {zero_input.shape}")
    print(f"   Output: {zero_output}")
    
    # Test with very small inputs
    small_input = np.ones((2, 4)) * 1e-6
    small_output = network.predict(small_input)
    print(f"\n2. Small input test:")
    print(f"   Input: 1e-6, shape {small_input.shape}")
    print(f"   Output range: [{np.min(small_output):.8f}, {np.max(small_output):.8f}]")
    
    # Test with large inputs
    large_input = np.ones((2, 4)) * 100
    large_output = network.predict(large_input)
    print(f"\n3. Large input test:")
    print(f"   Input: 100, shape {large_input.shape}")
    print(f"   Output range: [{np.min(large_output):.3f}, {np.max(large_output):.3f}]")
    
    # Test with single sample
    single_input = np.random.randn(4)  # 1D input
    single_output = network.predict(single_input)
    print(f"\n4. Single sample test:")
    print(f"   Input shape: {single_input.shape}")
    print(f"   Output shape: {single_output.shape}")
    
    return True


def test_anyonic_neural_networks():
    """Test anyonic neural network functionality if PyTorch is available."""
    if not TORCH_AVAILABLE:
        print("=== Testing Anyonic Neural Networks ===\n")
        print("Anyonic Neural Networks: Skipping (PyTorch not available)")
        return True
        
    print("=== Testing Anyonic Neural Networks ===\n")
    
    # Test different anyon types
    anyon_types = ["fibonacci", "ising", "generic"]
    test_input = np.random.randn(2, 4)
    
    print("Testing different anyon types:")
    for anyon_type in anyon_types:
        layer = AnyonicLayer(4, 3, anyon_type=anyon_type)
        output = layer.forward(test_input)
        print(f"  {anyon_type.capitalize()}: Input {test_input.shape} → Output {output.shape}")
        
        # Test deterministic behavior
        output2 = layer.forward(test_input)
        diff = np.abs(output - output2).max()
        print(f"    Deterministic: {diff < 1e-10} (max diff: {diff:.2e})")
    
    # Test braiding properties
    print("\nTesting braiding properties:")
    layer = AnyonicLayer(4, 2, anyon_type="fibonacci")
    
    # Test with basis vectors
    basis_vectors = [
        np.array([[1, 0, 0, 0]]),
        np.array([[0, 1, 0, 0]]),
        np.array([[1, 1, 0, 0]]),
    ]
    
    for i, vec in enumerate(basis_vectors):
        output = layer.forward(vec)
        print(f"  Basis {i+1}: {vec[0]} → {output[0]}")
    
    # Test network composition
    print("\nTesting anyonic network composition:")
    network = AlgebraicNeuralNetwork()
    network.add_layer(AnyonicLayer(4, 6, anyon_type="fibonacci"))
    network.add_layer(AnyonicLayer(6, 3, anyon_type="ising"))
    network.add_layer(AnyonicLayer(3, 2, anyon_type="generic"))
    
    test_input = np.random.randn(3, 4)
    output = network.predict(test_input)
    print(f"  Network: Input {test_input.shape} → Output {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    return True


def run_comprehensive_test():
    """Run all tests and report results."""
    print("Comprehensive Algebraic Neural Network Test Suite")
    print("="*60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Network Composition", test_network_composition), 
        ("Deterministic Behavior", test_deterministic_behavior),
        ("Mathematical Properties", test_mathematical_properties),
        ("Edge Cases", test_edge_cases),
        ("Anyonic Neural Networks", test_anyonic_neural_networks),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
            print(f"\n✓ {test_name}: PASSED")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n✗ {test_name}: FAILED - {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for test_name, result, error in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:.<30} {status}")
        if error:
            print(f"    Error: {error}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Algebraic Neural Network implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)