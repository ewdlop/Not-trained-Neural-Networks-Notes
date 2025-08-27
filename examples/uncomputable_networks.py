#!/usr/bin/env python3
"""
Uncomputable Neural Network Examples

This script demonstrates the various types of uncomputable neural network layers
and their theoretical foundations.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algebraic_neural_network import (
    HaltingOracleLayer, KolmogorovComplexityLayer, BusyBeaverLayer, 
    NonRecursiveLayer, create_uncomputable_network
)

def demonstrate_halting_oracle():
    """Demonstrate the Halting Oracle Layer."""
    print("=== Halting Oracle Layer Demonstration ===\n")
    
    print("The Halting Oracle Layer simulates access to a halting oracle,")
    print("which can theoretically answer whether a program halts on given input.")
    print("This is uncomputable in general, but we provide bounded approximations.\n")
    
    layer = HaltingOracleLayer(4, 3, max_iterations=1000)
    
    # Test with different "program" inputs
    test_cases = [
        ([1, 0, 0, 0], "Simple program"),
        ([0.5, 0.5, 0.5, 0.5], "Balanced program"),
        ([0.9, 0.9, 0.9, 0.9], "Complex program"),
        ([0.1, 0.1, 0.1, 0.1], "Minimal program"),
    ]
    
    for i, (input_data, description) in enumerate(test_cases, 1):
        input_array = np.array(input_data).reshape(1, -1)
        output = layer.forward(input_array)
        
        print(f"Test {i}: {description}")
        print(f"  Input: {input_data}")
        print(f"  Halting probabilities: {output[0]}")
        print(f"  Interpretation: {['Unlikely to halt' if p < 0.5 else 'Likely to halt' for p in output[0]]}")
        print()

def demonstrate_kolmogorov_complexity():
    """Demonstrate the Kolmogorov Complexity Layer."""
    print("=== Kolmogorov Complexity Layer Demonstration ===\n")
    
    print("The Kolmogorov Complexity Layer approximates the Kolmogorov complexity")
    print("of inputs - the length of the shortest program that outputs the input.")
    print("True Kolmogorov complexity is uncomputable, but we use compression heuristics.\n")
    
    layer = KolmogorovComplexityLayer(4, 3, precision=8)
    
    # Test with different complexity patterns
    test_cases = [
        ([1, 1, 1, 1], "Highly regular pattern"),
        ([1, 2, 3, 4], "Simple arithmetic sequence"), 
        ([0.123, 0.456, 0.789, 0.012], "Seemingly random"),
        ([1, 1, 2, 3], "Fibonacci-like pattern"),
    ]
    
    for i, (input_data, description) in enumerate(test_cases, 1):
        input_array = np.array(input_data).reshape(1, -1)
        output = layer.forward(input_array)
        
        print(f"Test {i}: {description}")
        print(f"  Input: {input_data}")
        print(f"  Complexity estimates: {output[0]}")
        print(f"  Average complexity: {np.mean(output[0]):.3f}")
        print()

def demonstrate_busy_beaver():
    """Demonstrate the Busy Beaver Layer."""
    print("=== Busy Beaver Layer Demonstration ===\n")
    
    print("The Busy Beaver Layer uses the Busy Beaver function BB(n),")
    print("which gives the maximum number of steps a halting n-state Turing machine can take.")
    print("BB(n) is uncomputable for general n, but known for small values.\n")
    
    layer = BusyBeaverLayer(4, 3)
    
    # Known BB values for reference
    print("Known Busy Beaver values:")
    print("  BB(1) = 1")
    print("  BB(2) = 4") 
    print("  BB(3) = 6")
    print("  BB(4) = 13")
    print("  BB(5) ≥ 4098")
    print()
    
    # Test with inputs that map to different BB parameters
    test_cases = [
        ([0.1, 0, 0, 0], "Maps to small machine"),
        ([0.5, 0.5, 0, 0], "Maps to medium machine"),
        ([1.0, 1.0, 1.0, 1.0], "Maps to larger machine"),
    ]
    
    for i, (input_data, description) in enumerate(test_cases, 1):
        input_array = np.array(input_data).reshape(1, -1)
        output = layer.forward(input_array)
        
        print(f"Test {i}: {description}")
        print(f"  Input: {input_data}")
        print(f"  Log BB values: {output[0]}")
        print(f"  Approximate BB values: {np.exp(output[0]) - 1}")
        print()

def demonstrate_non_recursive():
    """Demonstrate the Non-Recursive Layer."""
    print("=== Non-Recursive Layer Demonstration ===\n")
    
    print("The Non-Recursive Layer simulates operations on computably enumerable")
    print("but non-recursive sets. These are sets that can be enumerated but")
    print("for which membership cannot be decided algorithmically.\n")
    
    layer = NonRecursiveLayer(4, 3, enumeration_bound=1000)
    
    print(f"The layer simulates a c.e. set with {len(layer.ce_set)} enumerated elements")
    print("using the rule: numbers expressible as sum of two squares\n")
    
    # Test membership for different inputs
    test_cases = [
        ([1, 0, 0, 0], "Simple input"),
        ([2, 3, 5, 7], "Prime-like pattern"),
        ([1, 4, 9, 16], "Perfect squares"),
        ([0.5, 0.25, 0.125, 0.0625], "Geometric sequence"),
    ]
    
    for i, (input_data, description) in enumerate(test_cases, 1):
        input_array = np.array(input_data).reshape(1, -1)
        output = layer.forward(input_array)
        
        print(f"Test {i}: {description}")
        print(f"  Input: {input_data}")
        print(f"  Membership probabilities: {output[0]}")
        print(f"  Interpretations: {['Not enumerated' if p == 0.5 else ('In set' if p == 1.0 else 'Not in set') for p in output[0]]}")
        print()

def demonstrate_complete_network():
    """Demonstrate a complete uncomputable neural network."""
    print("=== Complete Uncomputable Neural Network ===\n")
    
    print("This demonstrates a full network combining all uncomputable layer types.")
    print("The network processes inputs through:")
    print("  1. Halting Oracle Layer (4→5)")
    print("  2. Kolmogorov Complexity Layer (5→4)")
    print("  3. Busy Beaver Layer (4→3)")
    print("  4. Non-Recursive Layer (3→2)")
    print()
    
    network = create_uncomputable_network()
    
    # Test with various input patterns
    test_cases = [
        "Random data",
        "Structured sequence", 
        "Constant values",
        "Alternating pattern"
    ]
    
    np.random.seed(42)
    test_inputs = [
        np.random.randn(4),
        np.array([1, 2, 3, 4]) / 4.0,
        np.ones(4) * 0.5,
        np.array([1, -1, 1, -1]) * 0.5
    ]
    
    for i, (input_data, description) in enumerate(zip(test_inputs, test_cases), 1):
        output = network.predict(input_data.reshape(1, -1))
        
        print(f"Test {i}: {description}")
        print(f"  Input: {input_data}")
        print(f"  Final output: {output[0]}")
        print(f"  Output interpretation: Decision values for uncomputable questions")
        print()
    
    # Demonstrate deterministic behavior
    print("Deterministic behavior verification:")
    test_input = np.random.randn(1, 4)
    output1 = network.predict(test_input)
    output2 = network.predict(test_input)
    difference = np.linalg.norm(output1 - output2)
    print(f"  Same input produces identical outputs: {difference < 1e-10}")
    print(f"  Difference: {difference:.2e}")

def main():
    """Run all uncomputable neural network demonstrations."""
    print("🔬 Uncomputable Neural Networks Examples")
    print("=" * 60)
    print("Exploring the theoretical boundaries of computation through")
    print("neural networks that incorporate uncomputable functions.\n")
    
    demonstrate_halting_oracle()
    print("\n" + "─" * 60 + "\n")
    
    demonstrate_kolmogorov_complexity()
    print("\n" + "─" * 60 + "\n")
    
    demonstrate_busy_beaver()
    print("\n" + "─" * 60 + "\n")
    
    demonstrate_non_recursive()
    print("\n" + "─" * 60 + "\n")
    
    demonstrate_complete_network()
    
    print("\n" + "=" * 60)
    print("✅ Uncomputable neural network demonstrations completed!")
    print("📚 See theory/uncomputable_networks.md for mathematical foundations.")

if __name__ == "__main__":
    main()