"""
Anyonic Neural Network Implementation

This example demonstrates neural networks using anyonic braiding operations
from topological quantum computing. Anyons are particles in 2D systems with
fractional exchange statistics that are neither fermionic nor bosonic.

The anyonic neural network implements braiding operations, topological charges,
and fusion rules without requiring any gradient-based training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from algebraic_neural_network import AnyonicLayer, AlgebraicNeuralNetwork, TORCH_AVAILABLE

if not TORCH_AVAILABLE:
    print("PyTorch is required for anyonic neural networks but is not available.")
    print("Please install PyTorch: pip install torch")
    sys.exit(1)

import torch


class AnyonicNeuralNetwork:
    """
    Specialized neural network using anyonic braiding operations.
    """
    
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        """
        Initialize anyonic neural network.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output
        """
        self.layers = []
        
        # Create layers with different anyon types
        anyon_types = ["fibonacci", "ising", "generic"]
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            anyon_type = anyon_types[i % len(anyon_types)]
            layer = AnyonicLayer(prev_dim, hidden_dim, anyon_type=anyon_type)
            self.layers.append(layer)
            prev_dim = hidden_dim
            
        # Final layer
        final_layer = AnyonicLayer(prev_dim, output_dim, anyon_type="fibonacci")
        self.layers.append(final_layer)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all anyonic layers."""
        current_input = x
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input


def test_basic_anyonic_operations():
    """Test basic anyonic braiding operations."""
    print("=== Anyonic Neural Networks: Basic Operations ===\n")
    
    # Create different types of anyonic layers
    fibonacci_layer = AnyonicLayer(4, 3, anyon_type="fibonacci")
    ising_layer = AnyonicLayer(4, 3, anyon_type="ising")
    generic_layer = AnyonicLayer(4, 3, anyon_type="generic")
    
    # Test input
    test_input = np.array([
        [1.0, 0.5, -0.3, 0.8],
        [0.2, -0.7, 1.2, -0.1],
        [-0.4, 0.9, 0.1, -0.6]
    ])
    
    print("Input data:")
    print(test_input)
    print(f"Input shape: {test_input.shape}\n")
    
    # Test different anyon types
    for name, layer in [("Fibonacci", fibonacci_layer), ("Ising", ising_layer), ("Generic", generic_layer)]:
        output = layer.forward(test_input)
        print(f"{name} Anyons:")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"  Output mean: {output.mean():.3f}")
        print(f"  Output std: {output.std():.3f}")
        print()


def test_anyonic_braiding_properties():
    """Test properties specific to anyonic braiding."""
    print("=== Anyonic Braiding Properties ===\n")
    
    layer = AnyonicLayer(4, 2, anyon_type="fibonacci")
    
    # Test with different inputs to show braiding effects
    inputs = [
        np.array([[1, 0, 0, 0]]),  # Basis vector e1
        np.array([[0, 1, 0, 0]]),  # Basis vector e2
        np.array([[0, 0, 1, 0]]),  # Basis vector e3
        np.array([[0, 0, 0, 1]]),  # Basis vector e4
        np.array([[1, 1, 0, 0]]),  # Combination
        np.array([[1, 0, 1, 0]]),  # Another combination
    ]
    
    print("Testing anyonic braiding on basis vectors:")
    for i, inp in enumerate(inputs):
        output = layer.forward(inp)
        print(f"Input {i+1}: {inp[0]} → Output: {output[0]}")
    
    print("\n" + "="*60)


def test_topological_properties():
    """Test topological properties of anyonic systems."""
    print("=== Topological Properties ===\n")
    
    layer = AnyonicLayer(6, 4, anyon_type="fibonacci")
    
    # Create inputs with different topological characteristics
    inputs = [
        np.random.normal(0, 0.1, (1, 6)),  # Small perturbation
        np.random.normal(0, 1.0, (1, 6)),  # Moderate values
        np.random.normal(0, 5.0, (1, 6)),  # Large values
        np.zeros((1, 6)),                   # Zero input
        np.ones((1, 6)),                    # Uniform input
    ]
    
    print("Testing topological charge behavior:")
    for i, inp in enumerate(inputs):
        output = layer.forward(inp)
        input_norm = np.linalg.norm(inp)
        output_norm = np.linalg.norm(output)
        
        print(f"Test {i+1}:")
        print(f"  Input norm: {input_norm:.3f}")
        print(f"  Output norm: {output_norm:.3f}")
        print(f"  Norm ratio: {output_norm/input_norm if input_norm > 0 else float('inf'):.3f}")
        print()


def test_anyonic_network_composition():
    """Test composition of multiple anyonic layers."""
    print("=== Anyonic Network Composition ===\n")
    
    # Create a multi-layer anyonic network
    network = AnyonicNeuralNetwork(
        input_dim=5,
        hidden_dims=[8, 6, 4],
        output_dim=2
    )
    
    # Test with various inputs
    test_cases = [
        ("Random normal", np.random.randn(3, 5)),
        ("Random uniform", np.random.uniform(-1, 1, (3, 5))),
        ("Sequential", np.arange(15).reshape(3, 5) / 10.0),
        ("Periodic", np.sin(np.arange(15).reshape(3, 5) * 0.5)),
    ]
    
    for name, test_input in test_cases:
        output = network.predict(test_input)
        print(f"{name} input:")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"  Output: {output}")
        print()


def test_deterministic_behavior():
    """Test that anyonic operations are deterministic."""
    print("=== Deterministic Behavior Test ===\n")
    
    layer = AnyonicLayer(4, 3, anyon_type="fibonacci")
    
    # Same input should produce same output
    test_input = np.random.randn(2, 4)
    
    output1 = layer.forward(test_input)
    output2 = layer.forward(test_input)
    output3 = layer.forward(test_input)
    
    diff_1_2 = np.abs(output1 - output2).max()
    diff_1_3 = np.abs(output1 - output3).max()
    diff_2_3 = np.abs(output2 - output3).max()
    
    print(f"Maximum differences between runs:")
    print(f"  Run 1 vs Run 2: {diff_1_2:.10f}")
    print(f"  Run 1 vs Run 3: {diff_1_3:.10f}")
    print(f"  Run 2 vs Run 3: {diff_2_3:.10f}")
    
    is_deterministic = diff_1_2 < 1e-10 and diff_1_3 < 1e-10 and diff_2_3 < 1e-10
    print(f"  Is deterministic: {is_deterministic}")
    
    if is_deterministic:
        print("✓ Anyonic operations are deterministic")
    else:
        print("✗ Anyonic operations are not deterministic")


def demonstrate_anyonic_neural_networks():
    """Main demonstration function."""
    print("Anyonic Neural Networks Demonstration")
    print("=" * 60)
    print()
    print("Anyonic neural networks use braiding operations from topological")
    print("quantum computing to process information without gradient-based training.")
    print()
    print("Key features:")
    print("- Braiding matrices from different anyon types (Fibonacci, Ising, etc.)")
    print("- Topological charge calculations")
    print("- Non-Abelian fusion rules")
    print("- Deterministic, training-free operation")
    print()
    print("=" * 60)
    print()
    
    # Run all tests
    test_basic_anyonic_operations()
    test_anyonic_braiding_properties()
    test_topological_properties()
    test_anyonic_network_composition()
    test_deterministic_behavior()
    
    print("\n" + "=" * 60)
    print("🎉 Anyonic Neural Network demonstration completed!")
    print("These networks showcase how topological quantum computing")
    print("concepts can be used for neural computation without training.")


if __name__ == "__main__":
    demonstrate_anyonic_neural_networks()