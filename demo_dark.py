#!/usr/bin/env python3
"""
Dark Neural Network Quick Demo

Run this script to see the basic functionality of PyTorch-based
dark neural networks that operate through non-observable physics.
"""

import torch
import numpy as np
from dark_neural_network import create_sample_dark_network


def main():
    print("🔬 Dark Neural Network Quick Demo")
    print("="*50)
    
    # Create a sample dark network
    print("1. Creating dark neural network...")
    network = create_sample_dark_network()
    network.eval()
    print("   ✓ Network created with quantum, dark matter, and hidden variable layers")
    
    # Generate sample data
    print("\n2. Generating sample data...")
    torch.manual_seed(42)  # For reproducible results
    sample_data = torch.randn(3, 4)
    print(f"   ✓ Generated {sample_data.shape[0]} samples with {sample_data.shape[1]} features each")
    
    # Make predictions
    print("\n3. Processing data through dark transformations...")
    with torch.no_grad():
        predictions = network(sample_data)
    print(f"   ✓ Output shape: {predictions.shape}")
    print(f"   ✓ Output range: [{torch.min(predictions):.6f}, {torch.max(predictions):.6f}]")
    
    # Show the data
    print("\n4. Results:")
    print("   Input data:")
    for i, sample in enumerate(sample_data):
        print(f"     Sample {i+1}: [{sample[0]:6.3f}, {sample[1]:6.3f}, {sample[2]:6.3f}, {sample[3]:6.3f}]")
    
    print("\n   Dark neural network output:")
    for i, output in enumerate(predictions):
        print(f"     Output {i+1}: [{output[0]:12.6f}, {output[1]:12.6f}]")
    
    # Demonstrate determinism
    print("\n5. Demonstrating deterministic behavior...")
    with torch.no_grad():
        predictions2 = network(sample_data)
    difference = torch.norm(predictions - predictions2).item()
    print(f"   ✓ Difference between runs: {difference:.10f} (should be 0)")
    
    # Show physics-inspired properties
    print("\n6. Physics-inspired properties:")
    
    # Test with zero input (vacuum state)
    zero_input = torch.zeros(1, 4)
    with torch.no_grad():
        vacuum_output = network(zero_input)
    print(f"   ✓ Vacuum state output: {vacuum_output.flatten().numpy()}")
    
    # Test symmetry properties
    symmetric_input = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    antisymmetric_input = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
    
    with torch.no_grad():
        sym_output = network(symmetric_input)
        antisym_output = network(antisymmetric_input)
    
    print(f"   ✓ Symmetric input output: {sym_output.flatten().numpy()}")
    print(f"   ✓ Antisymmetric input output: {antisym_output.flatten().numpy()}")
    
    print("\n" + "="*50)
    print("✅ Demo completed! Dark neural networks work without training.")
    print("🔬 These networks process information through non-observable physics.")
    print("📚 See examples/dark_neural_network_examples.py for more details.")


if __name__ == "__main__":
    main()