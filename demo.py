#!/usr/bin/env python3
"""
Quick demonstration of Algebraic Neural Networks

Run this script to see the basic functionality of algebraic neural networks.
"""

import numpy as np
from algebraic_neural_network import create_sample_network, create_uncomputable_network, create_non_algebraic_network

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
    
    # Bonus: Quick non-algebraic network demo
    print("\n🔬 Bonus: Non-Algebraic Neural Network Quick Demo")
    print("="*50)
    print("1. Creating non-algebraic neural network...")
    non_algebraic_network = create_non_algebraic_network()
    print("   ✓ Network created with probabilistic, chaos theory, information theory, and set theory layers")
    
    print("\n2. Processing same data through non-algebraic transformations...")
    non_algebraic_predictions = non_algebraic_network.predict(sample_data)
    print(f"   ✓ Output shape: {non_algebraic_predictions.shape}")
    print(f"   ✓ Output range: [{np.min(non_algebraic_predictions):.3f}, {np.max(non_algebraic_predictions):.3f}]")
    
    print("\n   Non-algebraic neural network output:")
    for i, output in enumerate(non_algebraic_predictions):
        print(f"     Output {i+1}: [{output[0]:8.3f}, {output[1]:8.3f}]")
    
    # Bonus: Quick uncomputable network demo
    print("\n🔬 Bonus: Uncomputable Neural Network Quick Demo")
    print("="*50)
    
    print("1. Creating uncomputable neural network...")
    uncomputable_network = create_uncomputable_network()
    print("   ✓ Network created with halting oracle, Kolmogorov complexity, Busy Beaver, and non-recursive layers")
    
    print("\n2. Processing same data through uncomputable transformations...")
    uncomputable_predictions = uncomputable_network.predict(sample_data)
    print(f"   ✓ Output shape: {uncomputable_predictions.shape}")
    print(f"   ✓ Output range: [{np.min(uncomputable_predictions):.3f}, {np.max(uncomputable_predictions):.3f}]")
    
    print("\n   Uncomputable neural network output:")
    for i, output in enumerate(uncomputable_predictions):
        print(f"     Output {i+1}: [{output[0]:6.3f}, {output[1]:6.3f}]")
    
    print("\n" + "="*50)
    print("🎯 All three networks operate without training but explore different mathematical domains!")
    print("📚 See theory/ and examples/ directories for more details.")

if __name__ == "__main__":
    main()