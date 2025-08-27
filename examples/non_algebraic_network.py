#!/usr/bin/env python3
"""
Non-Algebraic Neural Network Example

This example demonstrates non-algebraic neural networks that use probabilistic,
chaos theory, information theory, and set theory approaches instead of
traditional algebraic structures.

Author: Non-Algebraic Neural Network Research
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path to import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algebraic_neural_network import (
    ProbabilisticLayer, ChaosTheoryLayer, InformationTheoryLayer, SetTheoryLayer,
    AlgebraicNeuralNetwork, create_non_algebraic_network
)

def demonstrate_individual_layers():
    """Demonstrate each non-algebraic layer type individually."""
    print("=== Individual Non-Algebraic Layer Demonstrations ===\n")
    
    # Sample data
    np.random.seed(42)
    sample_data = np.random.randn(5, 4)
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Sample data:\n{sample_data}\n")
    
    # 1. Probabilistic Layer
    print("1. Probabilistic Layer (Gaussian Distribution):")
    prob_layer = ProbabilisticLayer(4, 3, distribution="gaussian")
    prob_output = prob_layer.forward(sample_data)
    print(f"   Output shape: {prob_output.shape}")
    print(f"   Output range: [{np.min(prob_output):.4f}, {np.max(prob_output):.4f}]")
    print(f"   Sample output: {prob_output[0]}")
    
    # 2. Chaos Theory Layer
    print("\n2. Chaos Theory Layer (Logistic Map):")
    chaos_layer = ChaosTheoryLayer(4, 3, map_type="logistic")
    chaos_output = chaos_layer.forward(sample_data)
    print(f"   Output shape: {chaos_output.shape}")
    print(f"   Output range: [{np.min(chaos_output):.4f}, {np.max(chaos_output):.4f}]")
    print(f"   Sample output: {chaos_output[0]}")
    
    # 3. Information Theory Layer
    print("\n3. Information Theory Layer (Entropy):")
    info_layer = InformationTheoryLayer(4, 3, info_type="entropy")
    info_output = info_layer.forward(sample_data)
    print(f"   Output shape: {info_output.shape}")
    print(f"   Output range: [{np.min(info_output):.4f}, {np.max(info_output):.4f}]")
    print(f"   Sample output: {info_output[0]}")
    
    # 4. Set Theory Layer
    print("\n4. Set Theory Layer (Membership Operations):")
    set_layer = SetTheoryLayer(4, 3, operation="membership")
    set_output = set_layer.forward(sample_data)
    print(f"   Output shape: {set_output.shape}")
    print(f"   Output range: [{np.min(set_output):.4f}, {np.max(set_output):.4f}]")
    print(f"   Sample output: {set_output[0]}")

def demonstrate_network_composition():
    """Demonstrate composition of non-algebraic layers into a complete network."""
    print("\n=== Non-Algebraic Network Composition ===\n")
    
    # Create network
    network = create_non_algebraic_network()
    print("Network structure:")
    print("  ProbabilisticLayer(4, 5) → ChaosTheoryLayer(5, 4) →")
    print("  InformationTheoryLayer(4, 3) → SetTheoryLayer(3, 2)")
    
    # Test with different data distributions
    test_cases = [
        ("Random Normal", np.random.randn(10, 4)),
        ("Random Uniform", np.random.uniform(-2, 2, (10, 4))),
        ("Structured Pattern", np.array([[i, i*2, i*3, i*4] for i in range(10)])),
    ]
    
    for case_name, test_data in test_cases:
        print(f"\n{case_name} Data:")
        output = network.predict(test_data)
        print(f"   Input shape: {test_data.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output mean: {np.mean(output):.4f}")
        print(f"   Output std: {np.std(output):.4f}")
        print(f"   Output range: [{np.min(output):.4f}, {np.max(output):.4f}]")

def demonstrate_deterministic_behavior():
    """Demonstrate that non-algebraic networks are deterministic."""
    print("\n=== Deterministic Behavior Test ===\n")
    
    network = create_non_algebraic_network()
    test_data = np.random.randn(5, 4)
    
    # Run multiple times
    outputs = []
    for i in range(3):
        output = network.predict(test_data)
        outputs.append(output)
        print(f"Run {i+1} output mean: {np.mean(output):.6f}")
    
    # Check consistency
    diff_1_2 = np.linalg.norm(outputs[0] - outputs[1])
    diff_1_3 = np.linalg.norm(outputs[0] - outputs[2])
    diff_2_3 = np.linalg.norm(outputs[1] - outputs[2])
    
    print(f"\nConsistency check:")
    print(f"   Run 1 vs 2 difference: {diff_1_2:.10f}")
    print(f"   Run 1 vs 3 difference: {diff_1_3:.10f}")
    print(f"   Run 2 vs 3 difference: {diff_2_3:.10f}")
    print(f"   Network is deterministic: {max(diff_1_2, diff_1_3, diff_2_3) < 1e-10}")

def compare_with_algebraic_networks():
    """Compare non-algebraic networks with algebraic approaches."""
    print("\n=== Comparison with Algebraic Networks ===\n")
    
    from algebraic_neural_network import create_sample_network
    
    # Same input data for fair comparison
    np.random.seed(123)
    test_data = np.random.randn(8, 4)
    
    # Create networks
    algebraic_net = create_sample_network()
    non_algebraic_net = create_non_algebraic_network()
    
    # Get outputs
    algebraic_output = algebraic_net.predict(test_data)
    non_algebraic_output = non_algebraic_net.predict(test_data)
    
    print("Network Comparison:")
    print(f"   Input shape: {test_data.shape}")
    print(f"   Algebraic output shape: {algebraic_output.shape}")
    print(f"   Non-algebraic output shape: {non_algebraic_output.shape}")
    
    print(f"\nAlgebraic Network:")
    print(f"   Output mean: {np.mean(algebraic_output):.4f}")
    print(f"   Output std: {np.std(algebraic_output):.4f}")
    print(f"   Output range: [{np.min(algebraic_output):.4f}, {np.max(algebraic_output):.4f}]")
    
    print(f"\nNon-Algebraic Network:")
    print(f"   Output mean: {np.mean(non_algebraic_output):.4f}")
    print(f"   Output std: {np.std(non_algebraic_output):.4f}")
    print(f"   Output range: [{np.min(non_algebraic_output):.4f}, {np.max(non_algebraic_output):.4f}]")

def visualize_chaos_behavior():
    """Visualize the chaotic behavior of the chaos theory layer."""
    print("\n=== Chaos Theory Layer Visualization ===\n")
    
    # Create chaos layer
    chaos_layer = ChaosTheoryLayer(1, 1, map_type="logistic")
    
    # Generate sequence of values using different initial conditions
    initial_conditions = np.linspace(0.1, 0.9, 9).reshape(-1, 1)
    
    print("Demonstrating sensitivity to initial conditions:")
    print("Initial Condition → Final Value (after 10 iterations)")
    
    for i, init_val in enumerate(initial_conditions):
        output = chaos_layer.forward(init_val)
        print(f"   {init_val[0]:.1f} → {output[0, 0]:.6f}")
    
    print("\nNote: Small changes in initial conditions lead to different outcomes,")
    print("demonstrating the chaotic nature of the transformation.")

def main():
    """Main demonstration function."""
    print("🔬 Non-Algebraic Neural Network Examples")
    print("=" * 60)
    
    demonstrate_individual_layers()
    demonstrate_network_composition()
    demonstrate_deterministic_behavior()
    compare_with_algebraic_networks()
    visualize_chaos_behavior()
    
    print("\n" + "=" * 60)
    print("✅ Non-algebraic neural network examples completed!")
    print("🔍 These networks demonstrate alternative approaches to")
    print("   information processing without traditional algebraic structures.")

if __name__ == "__main__":
    main()