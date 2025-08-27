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
    GeometricAlgebraLayer, HaltingOracleLayer, KolmogorovComplexityLayer,
    BusyBeaverLayer, NonRecursiveLayer, ProbabilisticLayer, ChaosTheoryLayer,
    InformationTheoryLayer, SetTheoryLayer, create_sample_network, 
    create_uncomputable_network, create_non_algebraic_network
)

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
    
    # Test non-algebraic layers
    print("\n4. Testing ProbabilisticLayer:")
    prob_layer = ProbabilisticLayer(4, 3, distribution="gaussian")
    prob_output = prob_layer.forward(test_input)
    print(f"   Input: {test_input.shape} → Output: {prob_output.shape}")
    print(f"   Output range: [{np.min(prob_output):.3f}, {np.max(prob_output):.3f}]")
    
    print("\n5. Testing ChaosTheoryLayer:")
    chaos_layer = ChaosTheoryLayer(4, 3, map_type="logistic")
    chaos_output = chaos_layer.forward(test_input)
    print(f"   Input: {test_input.shape} → Output: {chaos_output.shape}")
    print(f"   Output range: [{np.min(chaos_output):.3f}, {np.max(chaos_output):.3f}]")
    
    print("\n6. Testing InformationTheoryLayer:")
    info_layer = InformationTheoryLayer(4, 3, info_type="entropy")
    info_output = info_layer.forward(test_input)
    print(f"   Input: {test_input.shape} → Output: {info_output.shape}")
    print(f"   Output range: [{np.min(info_output):.3f}, {np.max(info_output):.3f}]")
    
    print("\n7. Testing SetTheoryLayer:")
    set_layer = SetTheoryLayer(4, 3, operation="membership")
    set_output = set_layer.forward(test_input)
    print(f"   Input: {test_input.shape} → Output: {set_output.shape}")
    print(f"   Output range: [{np.min(set_output):.3f}, {np.max(set_output):.3f}]")
    
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

def test_uncomputable_layers():
    """Test uncomputable neural network layers."""
    print("\n=== Testing Uncomputable Layers ===\n")
    
    test_input = np.random.randn(3, 4)
    
    # Test Halting Oracle Layer
    print("1. Testing HaltingOracleLayer:")
    halting_layer = HaltingOracleLayer(4, 3, max_iterations=100)
    halting_output = halting_layer.forward(test_input)
    print(f"   Input: {test_input.shape} → Output: {halting_output.shape}")
    print(f"   Output range: [{np.min(halting_output):.3f}, {np.max(halting_output):.3f}]")
    # Outputs should be probabilities between 0 and 1
    assert np.all(halting_output >= 0) and np.all(halting_output <= 1), "Halting oracle outputs must be in [0,1]"
    
    # Test Kolmogorov Complexity Layer
    print("\n2. Testing KolmogorovComplexityLayer:")
    kolmogorov_layer = KolmogorovComplexityLayer(4, 3, precision=6)
    kolmogorov_output = kolmogorov_layer.forward(test_input)
    print(f"   Input: {test_input.shape} → Output: {kolmogorov_output.shape}")
    print(f"   Output range: [{np.min(kolmogorov_output):.3f}, {np.max(kolmogorov_output):.3f}]")
    # Complexity should be non-negative
    assert np.all(kolmogorov_output >= 0), "Kolmogorov complexity must be non-negative"
    
    # Test Busy Beaver Layer
    print("\n3. Testing BusyBeaverLayer:")
    bb_layer = BusyBeaverLayer(4, 3)
    bb_output = bb_layer.forward(test_input)
    print(f"   Input: {test_input.shape} → Output: {bb_output.shape}")
    print(f"   Output range: [{np.min(bb_output):.3f}, {np.max(bb_output):.3f}]")
    # BB values should be positive
    assert np.all(bb_output > 0), "Busy Beaver values must be positive"
    
    # Test Non-Recursive Layer
    print("\n4. Testing NonRecursiveLayer:")
    nr_layer = NonRecursiveLayer(4, 3, enumeration_bound=100)
    nr_output = nr_layer.forward(test_input)
    print(f"   Input: {test_input.shape} → Output: {nr_output.shape}")
    print(f"   Output range: [{np.min(nr_output):.3f}, {np.max(nr_output):.3f}]")
    # Membership values should be in [0,1]
    assert np.all(nr_output >= 0) and np.all(nr_output <= 1), "Membership values must be in [0,1]"
    
    # Test deterministic behavior of uncomputable layers
    print("\n5. Testing deterministic behavior:")
    halting_output2 = halting_layer.forward(test_input)
    diff = np.linalg.norm(halting_output - halting_output2)
    print(f"   Determinism check: difference = {diff:.10f}")
    assert diff < 1e-10, "Uncomputable layers must be deterministic"
    
    return True

def test_uncomputable_network_composition():
    """Test composition of uncomputable neural network."""
    print("\n=== Testing Uncomputable Network Composition ===\n")
    
    # Create uncomputable network
    network = create_uncomputable_network()
    
    # Test with different input sizes
    test_cases = [
        (1, 4),    # Single sample
        (5, 4),    # Multiple samples
        (10, 4)    # Larger batch
    ]
    
    for i, (batch_size, input_size) in enumerate(test_cases, 1):
        test_input = np.random.randn(batch_size, input_size)
        output = network.predict(test_input)
        
        print(f"Test case {i}:")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output mean: {np.mean(output):.4f}")
        print(f"   Output std: {np.std(output):.4f}")
    
    return True

def test_non_algebraic_network_composition():
    """Test composition of non-algebraic neural network."""
    print("\n=== Testing Non-Algebraic Network Composition ===\n")
    
    # Create non-algebraic network
    network = create_non_algebraic_network()
    
    # Test with different input sizes
    test_cases = [
        (1, 4),    # Single sample
        (5, 4),    # Multiple samples
        (10, 4)    # Larger batch
    ]
    
    for i, (batch_size, input_size) in enumerate(test_cases, 1):
        test_input = np.random.randn(batch_size, input_size)
        output = network.predict(test_input)
        
        print(f"Test case {i}:")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output mean: {np.mean(output):.4f}")
        print(f"   Output std: {np.std(output):.4f}")
    
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

def run_comprehensive_test():
    """Run all tests and report results."""
    print("Comprehensive Algebraic Neural Network Test Suite")
    print("="*60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Network Composition", test_network_composition), 
        ("Deterministic Behavior", test_deterministic_behavior),
        ("Mathematical Properties", test_mathematical_properties),
        ("Uncomputable Layers", test_uncomputable_layers),
        ("Uncomputable Network Composition", test_uncomputable_network_composition),
        ("Non-Algebraic Network Composition", test_non_algebraic_network_composition),
        ("Edge Cases", test_edge_cases),
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