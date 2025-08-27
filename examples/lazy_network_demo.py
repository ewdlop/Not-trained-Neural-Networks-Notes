#!/usr/bin/env python3
"""
Example demonstrating Lazy Neural Networks

This example shows how lazy neural networks provide memory efficiency
and computational optimization through deferred evaluation.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lazy_neural_network import (
    LazyNeuralNetwork, LazyPolynomialLayer, LazyGroupTheoryLayer,
    LazyGeometricAlgebraLayer, create_sample_lazy_network
)


def create_large_lazy_network():
    """Create a larger lazy network for performance demonstration."""
    network = LazyNeuralNetwork()
    
    # Build a deeper network with compatible dimensions
    network.add_layer(LazyPolynomialLayer(10, 12, degree=3))
    network.add_layer(LazyGroupTheoryLayer(12, 10, group_order=8))
    network.add_layer(LazyGeometricAlgebraLayer(10, 8))
    network.add_layer(LazyPolynomialLayer(8, 6, degree=2))
    network.add_layer(LazyGroupTheoryLayer(6, 4, group_order=6))
    network.add_layer(LazyGeometricAlgebraLayer(4, 3))
    
    return network


def demonstrate_lazy_evaluation():
    """Demonstrate the core concept of lazy evaluation."""
    print("🧮 Lazy Neural Network - Core Concepts Demo")
    print("=" * 60)
    
    # Create a simple network
    network = create_sample_lazy_network()
    sample_input = np.random.randn(5, 4)
    
    print("\n1. LAZY EVALUATION CONCEPT")
    print("-" * 30)
    
    # Create lazy prediction without computing
    lazy_prediction = network.predict_lazy(sample_input)
    print(f"✓ Lazy prediction created: {type(lazy_prediction).__name__}")
    print(f"  - Computation deferred: {not lazy_prediction.is_computed}")
    print(f"  - Memory footprint: Minimal (no result stored yet)")
    
    # Now compute when needed
    print(f"\n💭 Computing result when needed...")
    start_time = time.time()
    result = lazy_prediction.compute()
    compute_time = time.time() - start_time
    
    print(f"✓ Result computed: shape {result.shape}")
    print(f"  - Time taken: {compute_time:.4f}s")
    print(f"  - Now cached: {lazy_prediction.is_computed}")
    
    # Subsequent accesses are instant
    start_time = time.time()
    result2 = lazy_prediction.value  # Using property accessor
    cached_time = time.time() - start_time
    
    print(f"✓ Cached access: {cached_time:.6f}s")
    print(f"  - Speed improvement: {compute_time/max(cached_time, 1e-6):.0f}x")


def demonstrate_memory_efficiency():
    """Demonstrate memory efficiency through lazy evaluation."""
    print("\n\n2. MEMORY EFFICIENCY DEMONSTRATION")
    print("-" * 40)
    
    network = create_large_lazy_network()
    
    # Create multiple lazy predictions without computing
    print("Creating 100 lazy predictions without computation...")
    lazy_predictions = []
    
    start_memory = 0  # Simplified memory tracking
    for i in range(100):
        large_input = np.random.randn(50, 10)
        lazy_pred = network.predict_lazy(large_input)
        lazy_predictions.append(lazy_pred)
    
    uncomputed_count = sum(1 for lp in lazy_predictions if not lp.is_computed)
    print(f"✓ Created {len(lazy_predictions)} lazy predictions")
    print(f"  - Uncomputed: {uncomputed_count}")
    print(f"  - Memory saved by not computing all results immediately")
    
    # Compute only a few when needed
    print(f"\nComputing only 10 out of 100 predictions...")
    computed_results = []
    for i in range(10):
        result = lazy_predictions[i].compute()
        computed_results.append(result)
    
    computed_count = sum(1 for lp in lazy_predictions if lp.is_computed)
    print(f"✓ Computed: {computed_count}/100 predictions")
    print(f"  - Memory efficiency: Only computed what was needed")


def demonstrate_partial_computation():
    """Demonstrate partial computation capabilities."""
    print("\n\n3. PARTIAL COMPUTATION DEMONSTRATION")
    print("-" * 45)
    
    network = create_large_lazy_network()
    sample_input = np.random.randn(10, 10)
    
    print(f"Network has {len(network.layers)} layers:")
    for i, layer in enumerate(network.layers):
        print(f"  Layer {i}: {layer.layer_type} ({layer.input_size} → {layer.output_size})")
    
    # Compute only specific layers
    print(f"\n🎯 Computing only layers [0, 2, 4] instead of all layers...")
    
    start_time = time.time()
    partial_results = network.partial_predict(sample_input, [0, 2, 4])
    partial_time = time.time() - start_time
    
    print(f"✓ Partial computation completed in {partial_time:.4f}s")
    for i, result in enumerate(partial_results):
        layer_idx = [0, 2, 4][i]
        print(f"  - Layer {layer_idx} output: {result.shape}")
    
    # Compare with full computation
    start_time = time.time()
    full_result = network.predict(sample_input)
    full_time = time.time() - start_time
    
    print(f"\n🔄 Full computation took {full_time:.4f}s")
    print(f"  - Final output: {full_result.shape}")
    print(f"  - Partial computation can be more efficient when only intermediate results are needed")


def demonstrate_caching_benefits():
    """Demonstrate caching benefits with repeated computations."""
    print("\n\n4. CACHING BENEFITS DEMONSTRATION")
    print("-" * 40)
    
    network = create_sample_lazy_network()
    
    # Create several different inputs
    inputs = [np.random.randn(5, 4) for _ in range(5)]
    
    print("Computing results for 5 different inputs...")
    
    # First round - no cache hits
    start_time = time.time()
    results_round1 = []
    for i, input_data in enumerate(inputs):
        result = network.predict(input_data)
        results_round1.append(result)
    first_round_time = time.time() - start_time
    
    stats_after_first = network.get_network_stats()
    print(f"✓ First round completed in {first_round_time:.4f}s")
    print(f"  - Cache hit rate: {stats_after_first['overall_cache_hit_rate']:.1%}")
    
    # Second round - should benefit from caching
    start_time = time.time()
    results_round2 = []
    for i, input_data in enumerate(inputs):
        result = network.predict(input_data)
        results_round2.append(result)
    second_round_time = time.time() - start_time
    
    stats_after_second = network.get_network_stats()
    print(f"✓ Second round completed in {second_round_time:.4f}s")
    print(f"  - Cache hit rate: {stats_after_second['overall_cache_hit_rate']:.1%}")
    print(f"  - Speed improvement: {first_round_time/max(second_round_time, 1e-6):.1f}x")
    
    # Verify results are identical
    total_diff = sum(np.linalg.norm(r1 - r2) for r1, r2 in zip(results_round1, results_round2))
    print(f"  - Result consistency: {total_diff:.2e} (should be ~0)")


def demonstrate_real_world_scenario():
    """Demonstrate a real-world scenario where lazy evaluation helps."""
    print("\n\n5. REAL-WORLD SCENARIO: CONDITIONAL PROCESSING")
    print("-" * 55)
    
    network = create_large_lazy_network()
    
    # Simulate a batch of data where we only process some based on conditions
    batch_size = 50
    data_batch = [np.random.randn(20, 10) for _ in range(batch_size)]
    
    print(f"Processing batch of {batch_size} samples...")
    print("Scenario: Only process samples that meet certain criteria")
    
    # Create lazy predictions for all samples
    lazy_predictions = []
    for data in data_batch:
        lazy_pred = network.predict_lazy(data)
        lazy_predictions.append(lazy_pred)
    
    print(f"✓ Created {len(lazy_predictions)} lazy predictions")
    
    # Simulate conditional processing - only compute for "interesting" samples
    # In practice, this might be based on metadata, user requests, etc.
    interesting_indices = np.random.choice(batch_size, size=10, replace=False)
    
    print(f"📊 Computing results only for {len(interesting_indices)} interesting samples...")
    
    start_time = time.time()
    processed_results = []
    for idx in interesting_indices:
        result = lazy_predictions[idx].compute()
        processed_results.append(result)
    conditional_time = time.time() - start_time
    
    print(f"✓ Conditional processing completed in {conditional_time:.4f}s")
    
    # Compare with processing all samples
    start_time = time.time()
    all_results = []
    for data in data_batch:
        result = network.predict(data)
        all_results.append(result)
    all_processing_time = time.time() - start_time
    
    print(f"🔄 Processing all samples took {all_processing_time:.4f}s")
    print(f"  - Savings: {((all_processing_time - conditional_time) / all_processing_time * 100):.1f}%")
    print(f"  - Efficiency gain: {all_processing_time/conditional_time:.1f}x")


def plot_performance_comparison():
    """Create a performance comparison plot."""
    print("\n\n6. PERFORMANCE VISUALIZATION")
    print("-" * 35)
    
    # Generate data for different batch sizes
    batch_sizes = [10, 20, 50, 100]
    lazy_times = []
    immediate_times = []
    cache_hit_rates = []
    
    network = create_sample_lazy_network()
    
    for batch_size in batch_sizes:
        print(f"Testing batch size {batch_size}...")
        
        # Reset network state
        network.reset_caches()
        
        # Generate test data
        test_data = [np.random.randn(5, 4) for _ in range(batch_size)]
        
        # Test lazy evaluation with caching
        start_time = time.time()
        for data in test_data:
            for _ in range(2):  # Process each sample twice to test caching
                lazy_result = network.predict_lazy(data)
                lazy_result.compute()
        lazy_time = time.time() - start_time
        lazy_times.append(lazy_time)
        
        stats = network.get_network_stats()
        cache_hit_rates.append(stats['overall_cache_hit_rate'])
        
        # Reset for fair comparison
        network.reset_caches()
        
        # Test immediate evaluation
        start_time = time.time()
        for data in test_data:
            for _ in range(2):
                network.predict(data)
        immediate_time = time.time() - start_time
        immediate_times.append(immediate_time)
    
    # Create plots
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Performance comparison
        ax1.plot(batch_sizes, lazy_times, 'o-', label='Lazy (with caching)', color='blue')
        ax1.plot(batch_sizes, immediate_times, 's-', label='Immediate', color='red')
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Performance: Lazy vs Immediate Evaluation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cache hit rates
        ax2.bar(batch_sizes, [rate * 100 for rate in cache_hit_rates], 
                color='green', alpha=0.7)
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Cache Hit Rate (%)')
        ax2.set_title('Cache Efficiency')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('lazy_neural_network_performance.png', dpi=150, bbox_inches='tight')
        print("✓ Performance plots saved as 'lazy_neural_network_performance.png'")
        
    except Exception as e:
        print(f"⚠️ Could not create plots: {e}")
    
    # Print summary
    print(f"\nPerformance Summary:")
    for i, batch_size in enumerate(batch_sizes):
        efficiency = immediate_times[i] / lazy_times[i]
        print(f"  Batch {batch_size}: {efficiency:.1f}x efficiency, {cache_hit_rates[i]:.1%} cache hits")


def main():
    """Run all lazy neural network demonstrations."""
    print("🚀 LAZY NEURAL NETWORKS - COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    print("Showcasing deferred computation, memory efficiency, and caching benefits")
    
    try:
        demonstrate_lazy_evaluation()
        demonstrate_memory_efficiency()
        demonstrate_partial_computation()
        demonstrate_caching_benefits()
        demonstrate_real_world_scenario()
        plot_performance_comparison()
        
        print("\n\n" + "=" * 80)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("Key benefits of Lazy Neural Networks:")
        print("  • 🎯 Deferred computation - compute only when needed")
        print("  • 💾 Memory efficiency - avoid storing unnecessary results")
        print("  • ⚡ Caching benefits - reuse computed results")
        print("  • 🧩 Partial computation - compute only required layers")
        print("  • 🔄 Conditional processing - process based on runtime conditions")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()