#!/usr/bin/env python3
"""
Quick demonstration of Algebraic Neural Networks

Run this script to see the basic functionality of algebraic neural networks.
"""

import numpy as np
from algebraic_neural_network import create_sample_network

def main():
    print("🧮 Algebraic Neural Network Quick Demo")
    print("="*50)
    
    # Create a sample network
    print("1. Creating algebraic neural network...")
    network = create_sample_network()
    print("   ✓ Network created with polynomial, group theory, and geometric algebra layers")
    
    # Generate sample data
    print("\n2. Generating sample data...")
    np.random.seed(42)  # For reproducible results
    sample_data = np.random.randn(3, 4)
    print(f"   ✓ Generated {sample_data.shape[0]} samples with {sample_data.shape[1]} features each")
    
    # Make predictions
    print("\n3. Processing data through algebraic transformations...")
    predictions = network.predict(sample_data)
    print(f"   ✓ Output shape: {predictions.shape}")
    print(f"   ✓ Output range: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")
    
    # Show the data
    print("\n4. Results:")
    print("   Input data:")
    for i, sample in enumerate(sample_data):
        print(f"     Sample {i+1}: [{sample[0]:6.3f}, {sample[1]:6.3f}, {sample[2]:6.3f}, {sample[3]:6.3f}]")
    
    print("\n   Algebraic neural network output:")
    for i, output in enumerate(predictions):
        print(f"     Output {i+1}: [{output[0]:8.3f}, {output[1]:8.3f}]")
    
    # Demonstrate determinism
    print("\n5. Demonstrating deterministic behavior...")
    predictions2 = network.predict(sample_data)
    difference = np.linalg.norm(predictions - predictions2)
    print(f"   ✓ Difference between runs: {difference:.10f} (should be 0)")
    
    print("\n" + "="*50)
    print("✅ Demo completed! Algebraic neural networks work without training.")
    print("📚 See theory/ and examples/ directories for more details.")

if __name__ == "__main__":
    main()