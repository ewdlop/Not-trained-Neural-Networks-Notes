#!/usr/bin/env python3
"""
Dark Neural Network Examples

This script demonstrates various dark neural network configurations
inspired by non-observable physics phenomena.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dark_neural_network import (
    DarkNeuralNetwork, QuantumDarkLayer, DarkMatterLayer, 
    HiddenVariableLayer, create_sample_dark_network
)


def example_quantum_superposition():
    """Demonstrate quantum-inspired superposition in dark layers."""
    print("=== Quantum Superposition Example ===\n")
    
    # Create quantum dark layer
    quantum_layer = QuantumDarkLayer(input_size=3, output_size=4, hidden_dim=8)
    quantum_layer.eval()
    
    # Test with orthogonal basis vectors
    basis_vectors = torch.tensor([
        [1.0, 0.0, 0.0],  # e1
        [0.0, 1.0, 0.0],  # e2
        [0.0, 0.0, 1.0],  # e3
        [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]  # superposition
    ], dtype=torch.float32)
    
    print("Input basis vectors:")
    for i, vec in enumerate(basis_vectors):
        print(f"  Vector {i+1}: {vec.numpy()}")
    
    with torch.no_grad():
        quantum_outputs = quantum_layer(basis_vectors)
    
    print("\nQuantum layer outputs (after hidden space processing):")
    for i, output in enumerate(quantum_outputs):
        print(f"  Output {i+1}: {output.numpy()}")
    
    # Analyze superposition properties
    print("\nSuperposition analysis:")
    superposition_output = quantum_outputs[3]
    individual_sum = quantum_outputs[0] + quantum_outputs[1] + quantum_outputs[2]
    
    print(f"Superposition output: {superposition_output.numpy()}")
    print(f"Sum of individuals: {individual_sum.numpy()}")
    print(f"Quantum interference: {(superposition_output - individual_sum/3).numpy()}")


def example_dark_matter_interactions():
    """Demonstrate dark matter inspired gravitational interactions."""
    print("\n=== Dark Matter Gravitational Interactions ===\n")
    
    # Create dark matter layer
    dark_layer = DarkMatterLayer(input_size=4, output_size=3, hidden_dim=6)
    dark_layer.eval()
    
    # Test with different "mass" distributions
    light_matter = torch.tensor([[0.1, 0.1, 0.1, 0.1]], dtype=torch.float32)
    heavy_matter = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    mixed_matter = torch.tensor([[0.1, 1.0, 0.1, 1.0]], dtype=torch.float32)
    
    test_inputs = torch.cat([light_matter, heavy_matter, mixed_matter], dim=0)
    
    print("Test matter distributions:")
    print(f"  Light matter: {light_matter.numpy().flatten()}")
    print(f"  Heavy matter: {heavy_matter.numpy().flatten()}")
    print(f"  Mixed matter: {mixed_matter.numpy().flatten()}")
    
    with torch.no_grad():
        dark_outputs = dark_layer(test_inputs)
    
    print("\nDark matter gravitational outputs:")
    for i, output in enumerate(dark_outputs):
        matter_type = ["Light", "Heavy", "Mixed"][i]
        print(f"  {matter_type} matter: {output.numpy()}")
    
    # Analyze gravitational scaling
    print("\nGravitational scaling analysis:")
    light_norm = torch.norm(dark_outputs[0]).item()
    heavy_norm = torch.norm(dark_outputs[1]).item()
    scaling_factor = heavy_norm / light_norm
    print(f"Heavy/Light output ratio: {scaling_factor:.3f}")


def example_hidden_variable_correlations():
    """Demonstrate hidden variable correlations and Bell-like effects."""
    print("\n=== Hidden Variable Correlations ===\n")
    
    # Create hidden variable layer
    hidden_layer = HiddenVariableLayer(input_size=2, output_size=4, hidden_dim=8)
    hidden_layer.eval()
    
    # Test with correlated and anti-correlated inputs
    correlated_pairs = torch.tensor([
        [1.0, 1.0],
        [-1.0, -1.0],
        [0.5, 0.5],
        [-0.5, -0.5]
    ], dtype=torch.float32)
    
    anti_correlated_pairs = torch.tensor([
        [1.0, -1.0],
        [-1.0, 1.0],
        [0.5, -0.5],
        [-0.5, 0.5]
    ], dtype=torch.float32)
    
    print("Testing correlated input pairs:")
    with torch.no_grad():
        corr_outputs = hidden_layer(correlated_pairs)
    
    for i, output in enumerate(corr_outputs):
        print(f"  Input {correlated_pairs[i].numpy()} → Output {output.numpy()}")
    
    print("\nTesting anti-correlated input pairs:")
    with torch.no_grad():
        anti_corr_outputs = hidden_layer(anti_correlated_pairs)
    
    for i, output in enumerate(anti_corr_outputs):
        print(f"  Input {anti_correlated_pairs[i].numpy()} → Output {output.numpy()}")
    
    # Analyze correlation patterns
    print("\nCorrelation analysis:")
    corr_variance = torch.var(corr_outputs, dim=0)
    anti_corr_variance = torch.var(anti_corr_outputs, dim=0)
    
    print(f"Correlated variance: {corr_variance.numpy()}")
    print(f"Anti-correlated variance: {anti_corr_variance.numpy()}")


def example_dark_network_composition():
    """Demonstrate composition of multiple dark layers."""
    print("\n=== Dark Network Composition ===\n")
    
    # Create a complex dark network
    network = DarkNeuralNetwork()
    
    # Add multiple dark layers
    network.add_layer(QuantumDarkLayer(input_size=5, output_size=8, hidden_dim=12))
    network.add_layer(DarkMatterLayer(input_size=8, output_size=6, hidden_dim=10))
    network.add_layer(HiddenVariableLayer(input_size=6, output_size=3, hidden_dim=8))
    
    network.eval()
    
    # Test with various input patterns
    test_patterns = torch.tensor([
        [1, 0, 0, 0, 0],      # Sparse
        [1, 1, 1, 1, 1],      # Uniform
        [1, -1, 1, -1, 1],    # Alternating
        [0.1, 0.2, 0.3, 0.4, 0.5],  # Gradient
        [0, 0, 0, 0, 0]       # Zero
    ], dtype=torch.float32)
    
    print("Testing dark network with various patterns:")
    
    with torch.no_grad():
        final_outputs = network(test_patterns)
    
    pattern_names = ["Sparse", "Uniform", "Alternating", "Gradient", "Zero"]
    
    for i, (pattern, output) in enumerate(zip(test_patterns, final_outputs)):
        print(f"  {pattern_names[i]:>10} input {pattern.numpy()} → {output.numpy()}")
    
    # Test network determinism
    print("\nTesting deterministic behavior across multiple runs:")
    with torch.no_grad():
        output1 = network(test_patterns)
        output2 = network(test_patterns)
        output3 = network(test_patterns)
    
    diff_12 = torch.norm(output1 - output2).item()
    diff_13 = torch.norm(output1 - output3).item()
    diff_23 = torch.norm(output2 - output3).item()
    
    print(f"Run 1 vs Run 2 difference: {diff_12:.10f}")
    print(f"Run 1 vs Run 3 difference: {diff_13:.10f}")
    print(f"Run 2 vs Run 3 difference: {diff_23:.10f}")
    
    is_deterministic = all(diff < 1e-10 for diff in [diff_12, diff_13, diff_23])
    print(f"Network is deterministic: {is_deterministic}")


def example_physics_inspired_processing():
    """Demonstrate physics-inspired data processing."""
    print("\n=== Physics-Inspired Data Processing ===\n")
    
    # Simulate particle physics data
    print("Simulating particle collision data processing:")
    
    # Create data resembling particle detector readings
    particle_data = torch.tensor([
        [10.5, 2.3, 0.8, 150.2],   # High energy particle
        [0.5, 0.1, 0.05, 15.1],    # Low energy particle
        [5.2, 1.1, 0.4, 75.3],     # Medium energy particle
        [0.0, 0.0, 0.0, 0.0],      # No detection
        [20.8, 4.6, 1.6, 300.4]    # Very high energy particle
    ], dtype=torch.float32)
    
    # Use dark network to process "unobservable" dark sector interactions
    dark_detector = DarkNeuralNetwork()
    dark_detector.add_layer(QuantumDarkLayer(4, 6, hidden_dim=10))
    dark_detector.add_layer(DarkMatterLayer(6, 4, hidden_dim=8))
    dark_detector.add_layer(HiddenVariableLayer(4, 2, hidden_dim=6))
    dark_detector.eval()
    
    print("Particle detector readings (Energy, Momentum_x, Momentum_y, Time):")
    for i, data in enumerate(particle_data):
        print(f"  Event {i+1}: {data.numpy()}")
    
    with torch.no_grad():
        dark_signatures = dark_detector(particle_data)
    
    print("\nDark sector signatures (non-observable physics):")
    for i, signature in enumerate(dark_signatures):
        print(f"  Event {i+1}: {signature.numpy()}")
    
    # Analyze dark signatures
    print("\nDark signature analysis:")
    high_energy_events = [0, 4]  # Events with high energy
    low_energy_events = [1, 3]   # Events with low energy
    
    high_energy_sigs = dark_signatures[high_energy_events]
    low_energy_sigs = dark_signatures[low_energy_events]
    
    high_energy_mean = torch.mean(high_energy_sigs, dim=0)
    low_energy_mean = torch.mean(low_energy_sigs, dim=0)
    
    print(f"High energy dark signature: {high_energy_mean.numpy()}")
    print(f"Low energy dark signature: {low_energy_mean.numpy()}")
    print(f"Dark signature difference: {(high_energy_mean - low_energy_mean).numpy()}")


def main():
    """Run all dark neural network examples."""
    print("Dark Neural Network Examples")
    print("=" * 60)
    print("Demonstrating non-observable physics inspired neural networks\n")
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run examples
    example_quantum_superposition()
    example_dark_matter_interactions()
    example_hidden_variable_correlations()
    example_dark_network_composition()
    example_physics_inspired_processing()
    
    print("\n" + "=" * 60)
    print("✅ All dark neural network examples completed!")
    print("🔬 These demonstrate non-observable physics in neural computation.")


if __name__ == "__main__":
    main()