#!/usr/bin/env python3
"""
Comprehensive test suite for Lazy Neural Networks

This script tests all components of the lazy neural network implementation
to ensure lazy evaluation works correctly.
"""

import sys
import os
import numpy as np
import time

# Add the parent directory to the path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all our implementations
from lazy_neural_network import (
    LazyValue, LazyCache, LazyLayer, LazyPolynomialLayer, 
    LazyGroupTheoryLayer, LazyGeometricAlgebraLayer, LazyNeuralNetwork,
    create_sample_lazy_network, lazy_computation
)


def test_lazy_value():
    """Test basic LazyValue functionality."""
    print("=== Testing LazyValue ===\n")
    
    # Test basic lazy computation
    def expensive_computation(x, y):
        time.sleep(0.01)  # Simulate expensive computation
        return x * y + 42
    
    lazy_val = LazyValue(expensive_computation, 5, 10)
    
    print(f"1. Initial state:")
    print(f"   Is computed: {lazy_val.is_computed}")
    print(f"   Computation count: {lazy_val.computation_count}")
    
    # First computation
    start_time = time.time()
    result1 = lazy_val.compute()
    first_time = time.time() - start_time
    
    print(f"\n2. After first computation:")
    print(f"   Result: {result1}")
    print(f"   Is computed: {lazy_val.is_computed}")
    print(f"   Computation count: {lazy_val.computation_count}")
    print(f"   Time taken: {first_time:.4f}s")
    
    # Second computation (should be cached)
    start_time = time.time()
    result2 = lazy_val.compute()
    second_time = time.time() - start_time
    
    print(f"\n3. After second computation:")
    print(f"   Result: {result2}")
    print(f"   Computation count: {lazy_val.computation_count}")
    print(f"   Time taken: {second_time:.4f}s")
    print(f"   Speed improvement: {first_time/max(second_time, 1e-6):.1f}x")
    
    # Test reset
    lazy_val.reset()
    print(f"\n4. After reset:")
    print(f"   Is computed: {lazy_val.is_computed}")
    
    assert result1 == result2 == 92
    assert lazy_val.computation_count == 1  # Reset doesn't change count
    return True


def test_lazy_cache():
    """Test LazyCache functionality."""
    print("\n=== Testing LazyCache ===\n")
    
    cache = LazyCache(max_size=3)
    
    # Create some lazy values
    lazy_vals = []
    for i in range(5):
        lazy_val = LazyValue(lambda x=i: x ** 2)
        lazy_vals.append(lazy_val)
        cache.put(f"key_{i}", lazy_val)
    
    print(f"1. Cache after adding 5 items (max_size=3):")
    print(f"   Cache size: {cache.size()}")
    
    # Test cache hits and misses
    hit_val = cache.get("key_3")  # Should be a hit
    miss_val = cache.get("key_0")  # Should be a miss (evicted)
    
    print(f"2. Cache lookup results:")
    print(f"   key_3 (should hit): {'HIT' if hit_val else 'MISS'}")
    print(f"   key_0 (should miss): {'HIT' if miss_val else 'MISS'}")
    
    assert cache.size() == 3
    assert hit_val is not None
    assert miss_val is None
    return True


def test_lazy_layers():
    """Test individual lazy layer functionality."""
    print("\n=== Testing Lazy Layers ===\n")
    
    # Test data
    test_input = np.random.randn(3, 4)
    
    # Test LazyPolynomialLayer
    print("1. Testing LazyPolynomialLayer:")
    poly_layer = LazyPolynomialLayer(4, 3, degree=2)
    
    # Test lazy evaluation
    lazy_output = poly_layer.forward_lazy(test_input)
    print(f"   Lazy output created: {lazy_output}")
    print(f"   Is computed: {lazy_output.is_computed}")
    
    # Compute result
    output = lazy_output.compute()
    print(f"   After computation: {lazy_output.is_computed}")
    print(f"   Output shape: {output.shape}")
    
    # Test immediate evaluation gives same result
    immediate_output = poly_layer.forward(test_input)
    diff = np.linalg.norm(output - immediate_output)
    print(f"   Difference lazy vs immediate: {diff:.10f}")
    
    # Test LazyGroupTheoryLayer
    print("\n2. Testing LazyGroupTheoryLayer:")
    group_layer = LazyGroupTheoryLayer(4, 3, group_order=6)
    
    lazy_group_output = group_layer.forward_lazy(test_input)
    group_output = lazy_group_output.compute()
    immediate_group_output = group_layer.forward(test_input)
    
    group_diff = np.linalg.norm(group_output - immediate_group_output)
    print(f"   Output shape: {group_output.shape}")
    print(f"   Difference lazy vs immediate: {group_diff:.10f}")
    
    # Test LazyGeometricAlgebraLayer
    print("\n3. Testing LazyGeometricAlgebraLayer:")
    geo_layer = LazyGeometricAlgebraLayer(4, 3)
    
    lazy_geo_output = geo_layer.forward_lazy(test_input)
    geo_output = lazy_geo_output.compute()
    immediate_geo_output = geo_layer.forward(test_input)
    
    geo_diff = np.linalg.norm(geo_output - immediate_geo_output)
    print(f"   Output shape: {geo_output.shape}")
    print(f"   Difference lazy vs immediate: {geo_diff:.10f}")
    
    # Test caching
    print("\n4. Testing layer caching:")
    stats = poly_layer.get_stats()
    print(f"   Polynomial layer stats: {stats}")
    
    # Second call with same input should hit cache
    poly_layer.forward_lazy(test_input)
    updated_stats = poly_layer.get_stats()
    print(f"   Updated stats: {updated_stats}")
    
    assert diff < 1e-10
    assert group_diff < 1e-10
    assert geo_diff < 1e-10
    assert updated_stats['cache_hits'] > stats['cache_hits']
    return True


def test_lazy_network():
    """Test LazyNeuralNetwork functionality."""
    print("\n=== Testing LazyNeuralNetwork ===\n")
    
    # Create network
    network = create_sample_lazy_network()
    test_input = np.random.randn(4, 4)
    
    print(f"1. Network structure:")
    print(f"   Number of layers: {len(network.layers)}")
    for i, layer in enumerate(network.layers):
        print(f"   Layer {i}: {layer.layer_type} ({layer.input_size} → {layer.output_size})")
    
    # Test lazy prediction
    print(f"\n2. Testing lazy prediction:")
    lazy_result = network.predict_lazy(test_input)
    print(f"   Lazy result created: {lazy_result}")
    print(f"   Is computed: {lazy_result.is_computed}")
    
    output = lazy_result.compute()
    print(f"   After computation: {lazy_result.is_computed}")
    print(f"   Output shape: {output.shape}")
    
    # Test immediate prediction
    immediate_output = network.predict(test_input)
    network_diff = np.linalg.norm(output - immediate_output)
    print(f"   Difference lazy vs immediate: {network_diff:.10f}")
    
    # Test partial prediction
    print(f"\n3. Testing partial prediction:")
    partial_outputs = network.partial_predict(test_input[:2], [0, 2])
    print(f"   Partial outputs for layers [0, 2]:")
    for i, partial_out in enumerate(partial_outputs):
        print(f"     Output {i}: shape {partial_out.shape}")
    
    # Test statistics
    print(f"\n4. Network statistics:")
    stats = network.get_network_stats()
    print(f"   Network stats: {stats['network_stats']}")
    print(f"   Overall cache hit rate: {stats['overall_cache_hit_rate']:.2%}")
    
    assert network_diff < 1e-10
    assert len(partial_outputs) == 2
    return True


def test_performance_comparison():
    """Compare performance of lazy vs immediate evaluation."""
    print("\n=== Testing Performance Comparison ===\n")
    
    # Create networks
    lazy_network = create_sample_lazy_network()
    
    # Test data
    test_inputs = [np.random.randn(10, 4) for _ in range(5)]
    
    print("1. Performance test with repeated computations:")
    
    # Lazy evaluation with repeated inputs
    start_time = time.time()
    lazy_results = []
    for test_input in test_inputs:
        for _ in range(3):  # Repeat same input
            lazy_result = lazy_network.predict_lazy(test_input)
            lazy_results.append(lazy_result.compute())
    lazy_time = time.time() - start_time
    
    # Reset caches for fair comparison
    lazy_network.reset_caches()
    
    # Immediate evaluation
    start_time = time.time()
    immediate_results = []
    for test_input in test_inputs:
        for _ in range(3):  # Repeat same input
            immediate_results.append(lazy_network.predict(test_input))
    immediate_time = time.time() - start_time
    
    print(f"   Lazy evaluation time: {lazy_time:.4f}s")
    print(f"   Immediate evaluation time: {immediate_time:.4f}s")
    
    # Verify results are the same
    total_diff = 0
    for lazy_res, imm_res in zip(lazy_results, immediate_results):
        total_diff += np.linalg.norm(lazy_res - imm_res)
    
    print(f"   Total difference in results: {total_diff:.10f}")
    
    # Show final statistics
    final_stats = lazy_network.get_network_stats()
    print(f"\n2. Final network statistics:")
    print(f"   Total forward calls: {final_stats['network_stats']['forward_calls']}")
    print(f"   Cache hit rate: {final_stats['overall_cache_hit_rate']:.2%}")
    print(f"   Total cache size: {final_stats['total_cache_size']}")
    
    assert total_diff < 1e-8
    return True


def test_lazy_decorator():
    """Test the lazy_computation decorator."""
    print("\n=== Testing Lazy Decorator ===\n")
    
    @lazy_computation
    def fibonacci(n):
        """Compute fibonacci number (inefficiently for testing)."""
        if n <= 1:
            return n
        return fibonacci(n-1).compute() + fibonacci(n-2).compute()
    
    print("1. Testing lazy fibonacci:")
    
    # Compute fibonacci numbers
    fib_5 = fibonacci(5)
    print(f"   Fibonacci(5) lazy object created: {fib_5}")
    print(f"   Is computed: {fib_5.is_computed}")
    
    result = fib_5.compute()
    print(f"   Fibonacci(5) = {result}")
    print(f"   Is computed after evaluation: {fib_5.is_computed}")
    
    # Test multiple evaluations
    result2 = fib_5.compute()
    print(f"   Second evaluation: {result2}")
    print(f"   Computation count: {fib_5.computation_count}")
    
    assert result == result2 == 5
    assert fib_5.computation_count == 1
    return True


def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\n=== Testing Edge Cases ===\n")
    
    network = create_sample_lazy_network()
    
    # Test with zero input
    print("1. Zero input test:")
    zero_input = np.zeros((2, 4))
    zero_lazy = network.predict_lazy(zero_input)
    zero_result = zero_lazy.compute()
    print(f"   Zero input result: {zero_result}")
    
    # Test with single sample
    print("\n2. Single sample test:")
    single_input = np.random.randn(4)
    single_lazy = network.predict_lazy(single_input)
    single_result = single_lazy.compute()
    print(f"   Single sample shape: {single_result.shape}")
    
    # Test partial prediction with invalid indices
    print("\n3. Invalid layer indices test:")
    try:
        network.partial_predict(zero_input, [10])  # Invalid index
        print("   ERROR: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"   Correctly caught error: {e}")
    
    # Test empty partial prediction
    empty_result = network.partial_predict(zero_input, [])
    print(f"   Empty partial prediction: {empty_result}")
    
    assert len(empty_result) == 0
    return True


def run_lazy_comprehensive_test():
    """Run all lazy neural network tests and report results."""
    print("Comprehensive Lazy Neural Network Test Suite")
    print("="*70)
    
    tests = [
        ("LazyValue Functionality", test_lazy_value),
        ("LazyCache Functionality", test_lazy_cache),
        ("Lazy Layers", test_lazy_layers),
        ("Lazy Network", test_lazy_network),
        ("Performance Comparison", test_performance_comparison),
        ("Lazy Decorator", test_lazy_decorator),
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
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for test_name, result, error in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:.<40} {status}")
        if error:
            print(f"    Error: {error}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Lazy Neural Network implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = run_lazy_comprehensive_test()
    sys.exit(0 if success else 1)