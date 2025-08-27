"""
Supersymmetry-based Algebraic Neural Network

This example demonstrates neural networks that use supersymmetric algebra
from theoretical physics, including Grassmann algebra, anticommuting variables,
and bosonic/fermionic duality.
"""

import numpy as np
import sys
import os

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algebraic_neural_network import SupersymmetryLayer, AlgebraicNeuralNetwork


def test_grassmann_algebra():
    """Test fundamental Grassmann algebra properties."""
    print("=== Supersymmetric Neural Network: Grassmann Algebra ===\n")
    
    susy_layer = SupersymmetryLayer(4, 3, n_grassmann=3)
    
    # Create test Grassmann variables
    theta1 = np.array([1, 0, 0, 0])
    theta2 = np.array([0, 1, 0, 0])
    theta3 = np.array([0, 0, 1, 0])
    
    print("Testing Grassmann algebra properties:")
    print(f"θ₁ = {theta1}")
    print(f"θ₂ = {theta2}")
    print(f"θ₃ = {theta3}")
    
    # Test anticommutation relation: θᵢθⱼ = -θⱼθᵢ
    print("\n1. Anticommutation Relations:")
    
    for i, (name_i, theta_i) in enumerate([("θ₁", theta1), ("θ₂", theta2), ("θ₃", theta3)]):
        for j, (name_j, theta_j) in enumerate([("θ₁", theta1), ("θ₂", theta2), ("θ₃", theta3)]):
            if i < j:  # Avoid redundant tests
                prod_ij = susy_layer.grassmann_product(theta_i, theta_j)
                prod_ji = susy_layer.grassmann_product(theta_j, theta_i)
                print(f"   {name_i}∧{name_j} = {prod_ij:.4f}")
                print(f"   {name_j}∧{name_i} = {prod_ji:.4f}")
                print(f"   Sum (should be ≈0): {prod_ij + prod_ji:.4f}")
                print()
    
    # Test nilpotent property: θᵢ² = 0
    print("2. Nilpotent Property (θᵢ² = 0):")
    for name, theta in [("θ₁", theta1), ("θ₂", theta2), ("θ₃", theta3)]:
        squared = susy_layer.grassmann_product(theta, theta)
        print(f"   {name}² = {squared:.4f}")
    
    print()


def test_supersymmetric_transformations():
    """Test supersymmetric transformations and their properties."""
    print("=== Supersymmetric Transformations ===\n")
    
    susy_layer = SupersymmetryLayer(4, 4, n_grassmann=2)
    
    # Create bosonic and fermionic fields
    phi = np.array([1.0, 0.5, -0.3, 0.8])  # Bosonic field
    psi = np.array([0.2, -0.1, 0.4, -0.2]) # Fermionic field
    
    print("Original fields:")
    print(f"Bosonic field φ:  {phi}")
    print(f"Fermionic field ψ: {psi}")
    print(f"φ norm: {np.linalg.norm(phi):.4f}")
    print(f"ψ norm: {np.linalg.norm(psi):.4f}")
    
    # Apply multiple supersymmetric transformations
    print("\nApplying supersymmetric transformations:")
    
    current_phi, current_psi = phi.copy(), psi.copy()
    
    for i in range(3):
        new_phi, new_psi = susy_layer.supersymmetric_transform(current_phi, current_psi)
        
        print(f"\nTransformation {i+1}:")
        print(f"φ' = {new_phi}")
        print(f"ψ' = {new_psi}")
        
        # Compute changes
        delta_phi = new_phi - current_phi
        delta_psi = new_psi - current_psi
        
        print(f"Δφ = {delta_phi}")
        print(f"Δψ = {delta_psi}")
        print(f"Change magnitude: |Δφ| = {np.linalg.norm(delta_phi):.4f}, |Δψ| = {np.linalg.norm(delta_psi):.4f}")
        
        current_phi, current_psi = new_phi, new_psi
    
    print()


def test_bosonic_fermionic_duality():
    """Test the duality between bosonic and fermionic components."""
    print("=== Bosonic-Fermionic Duality ===\n")
    
    network = AlgebraicNeuralNetwork()
    network.add_layer(SupersymmetryLayer(6, 4, n_grassmann=2))
    network.add_layer(SupersymmetryLayer(4, 2, n_grassmann=2))
    
    # Create input with clear bosonic/fermionic structure
    # First half: bosonic (commuting), second half: fermionic (anticommuting)
    bosonic_input = np.array([1.0, 0.5, -0.2])
    fermionic_input = np.array([0.1, -0.3, 0.2])
    
    # Combine into full input
    full_input = np.concatenate([bosonic_input, fermionic_input])
    
    print("Input structure:")
    print(f"Bosonic part:   {bosonic_input}")
    print(f"Fermionic part: {fermionic_input}")
    print(f"Full input:     {full_input}")
    
    # Process through supersymmetric network
    output = network.predict(full_input.reshape(1, -1))
    
    print(f"\nNetwork output: {output[0]}")
    print(f"Output shape: {output.shape}")
    
    # Test with swapped bosonic/fermionic parts
    swapped_input = np.concatenate([fermionic_input, bosonic_input])
    swapped_output = network.predict(swapped_input.reshape(1, -1))
    
    print(f"\nWith swapped B/F parts:")
    print(f"Swapped input:  {swapped_input}")
    print(f"Swapped output: {swapped_output[0]}")
    
    # Compare outputs
    output_diff = np.linalg.norm(output - swapped_output)
    print(f"\nOutput difference: {output_diff:.4f}")
    print(f"Sensitivity to B/F ordering: {'High' if output_diff > 1.0 else 'Low'}")
    
    print()


def test_superspace_coordinates():
    """Test operations in superspace (ordinary + Grassmann coordinates)."""
    print("=== Superspace Coordinate Processing ===\n")
    
    susy_layer = SupersymmetryLayer(4, 3, n_grassmann=3)
    
    # Create superspace coordinates: (x, θ) where x are ordinary, θ are Grassmann
    ordinary_coords = np.array([1.0, 2.0])  # x, y coordinates
    grassmann_coords = np.array([0.1, 0.2])  # θ₁, θ₂ coordinates
    
    superspace_point = np.concatenate([ordinary_coords, grassmann_coords])
    
    print("Superspace coordinate processing:")
    print(f"Ordinary coordinates (x,y): {ordinary_coords}")
    print(f"Grassmann coordinates (θ₁,θ₂): {grassmann_coords}")
    print(f"Superspace point: {superspace_point}")
    
    # Process multiple superspace points
    superspace_batch = np.array([
        [1.0, 2.0, 0.1, 0.2],  # (x₁, y₁, θ₁, θ₂)
        [0.5, -1.0, -0.1, 0.3], # (x₂, y₂, θ₁, θ₂)
        [-0.5, 0.8, 0.2, -0.1], # (x₃, y₃, θ₁, θ₂)
    ])
    
    output = susy_layer.forward(superspace_batch)
    
    print(f"\nBatch processing results:")
    print(f"Input shape: {superspace_batch.shape}")
    print(f"Output shape: {output.shape}")
    
    for i, (inp, out) in enumerate(zip(superspace_batch, output)):
        print(f"Point {i+1}: {inp} → {out}")
    
    print()


def test_supersymmetric_network_composition():
    """Test composing multiple supersymmetric layers."""
    print("=== Supersymmetric Network Composition ===\n")
    
    # Create a deep supersymmetric network
    network = AlgebraicNeuralNetwork()
    network.add_layer(SupersymmetryLayer(6, 5, n_grassmann=2))
    network.add_layer(SupersymmetryLayer(5, 4, n_grassmann=3))
    network.add_layer(SupersymmetryLayer(4, 3, n_grassmann=2))
    network.add_layer(SupersymmetryLayer(3, 2, n_grassmann=1))
    
    print("Deep supersymmetric network architecture:")
    print("6 → 5 → 4 → 3 → 2 (with Grassmann algebra at each layer)")
    
    # Test with different types of inputs
    test_inputs = {
        "Random": np.random.randn(3, 6),
        "Symmetric": np.array([[1, 1, 1, 1, 1, 1], [-1, -1, -1, -1, -1, -1], [0, 0, 0, 0, 0, 0]]),
        "Structured": np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], [1, -1, 1, -1, 1, -1]])
    }
    
    for name, test_input in test_inputs.items():
        print(f"\n{name} input test:")
        print(f"Input:\n{test_input}")
        
        output = network.predict(test_input)
        print(f"Output:\n{output}")
        print(f"Output statistics: mean={np.mean(output):.4f}, std={np.std(output):.4f}")
        
        # Test determinism
        output2 = network.predict(test_input)
        determinism_check = np.allclose(output, output2)
        print(f"Deterministic: {determinism_check}")


if __name__ == "__main__":
    print("Supersymmetric Algebraic Neural Network Demo\n")
    print("=" * 60)
    
    # Run all supersymmetric tests
    test_grassmann_algebra()
    test_supersymmetric_transformations()
    test_bosonic_fermionic_duality()
    test_superspace_coordinates()
    test_supersymmetric_network_composition()
    
    print("\n" + "=" * 60)
    print("Supersymmetric neural network demo completed successfully!")
    print("\nKey physics concepts demonstrated:")
    print("• Grassmann algebra with anticommuting variables")
    print("• Nilpotent properties (θ² = 0)")
    print("• Supersymmetric field transformations")
    print("• Bosonic/fermionic duality")
    print("• Superspace coordinate processing")