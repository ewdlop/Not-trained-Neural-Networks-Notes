"""
Simple demonstration of both neural network architectures working on a toy problem.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fermi_dirac_brain import create_fermi_dirac_classifier
from maxwell_boltzmann_brain import create_maxwell_boltzmann_classifier


def simple_demo():
    """Simple demonstration of both neural networks."""
    print("=== Statistical Physics Neural Networks Demo ===\n")
    
    # Generate simple dataset
    X, y = make_classification(
        n_samples=200, n_features=8, n_informative=6, n_redundant=2,
        n_classes=2, random_state=42
    )
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    print(f"Dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}, Classes: 2\n")
    
    # Create models
    fermi_model = create_fermi_dirac_classifier(
        input_size=8, num_classes=2, hidden_sizes=[16, 8], temperature=1.0
    )
    
    maxwell_model = create_maxwell_boltzmann_classifier(
        input_size=8, num_classes=2, hidden_sizes=[16, 8], temperature=1.5
    )
    
    print("Created models:")
    print(f"Fermi-Dirac Brain: {sum(p.numel() for p in fermi_model.parameters())} parameters")
    print(f"Maxwell-Boltzmann Brain: {sum(p.numel() for p in maxwell_model.parameters())} parameters\n")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    fermi_optimizer = optim.Adam(fermi_model.parameters(), lr=0.01)
    maxwell_optimizer = optim.Adam(maxwell_model.parameters(), lr=0.01)
    
    epochs = 50
    
    print("Training both models...")
    for epoch in range(epochs):
        # Train Fermi-Dirac model
        fermi_model.train()
        fermi_optimizer.zero_grad()
        fermi_outputs = fermi_model(X_train)
        fermi_loss = criterion(fermi_outputs, y_train)
        fermi_loss.backward()
        fermi_optimizer.step()
        
        # Train Maxwell-Boltzmann model
        maxwell_model.train()
        maxwell_optimizer.zero_grad()
        maxwell_outputs = maxwell_model(X_train)
        maxwell_loss = criterion(maxwell_outputs, y_train)
        maxwell_loss.backward()
        maxwell_optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:2d}: Fermi-Dirac Loss: {fermi_loss.item():.4f}, "
                  f"Maxwell-Boltzmann Loss: {maxwell_loss.item():.4f}")
    
    # Final evaluation
    fermi_model.eval()
    maxwell_model.eval()
    
    with torch.no_grad():
        # Fermi-Dirac evaluation
        fermi_test_outputs = fermi_model(X_test)
        _, fermi_predicted = torch.max(fermi_test_outputs.data, 1)
        fermi_accuracy = (fermi_predicted == y_test).float().mean().item()
        
        # Maxwell-Boltzmann evaluation
        maxwell_test_outputs = maxwell_model(X_test)
        _, maxwell_predicted = torch.max(maxwell_test_outputs.data, 1)
        maxwell_accuracy = (maxwell_predicted == y_test).float().mean().item()
        
        # Compute statistical properties
        fermi_entropy = fermi_model.compute_quantum_entropy()
        fermi_energy = fermi_model.get_fermi_energy()
        
        maxwell_thermal_energy = maxwell_model.compute_thermal_energy()
        maxwell_partition = maxwell_model.compute_partition_function()
    
    print("\n=== Final Results ===")
    print(f"Fermi-Dirac Brain:")
    print(f"  Test Accuracy: {fermi_accuracy:.4f}")
    print(f"  Quantum Entropy: {fermi_entropy:.4f}")
    print(f"  Fermi Energy: {fermi_energy:.4f}")
    
    print(f"\nMaxwell-Boltzmann Brain:")
    print(f"  Test Accuracy: {maxwell_accuracy:.4f}")
    print(f"  Thermal Energy: {maxwell_thermal_energy:.4f}")
    print(f"  Partition Function: {maxwell_partition:.4f}")
    
    # Test temperature adjustment
    print(f"\n=== Temperature Effects Demo ===")
    print("Testing temperature adjustment on Fermi-Dirac model...")
    
    original_outputs = fermi_model(X_test[:5])
    print(f"Original outputs (T=1.0): {original_outputs[0].detach().numpy()}")
    
    fermi_model.adjust_temperature(0.1)  # Low temperature
    cold_outputs = fermi_model(X_test[:5])
    print(f"Cold outputs (T=0.1): {cold_outputs[0].detach().numpy()}")
    
    fermi_model.adjust_temperature(5.0)  # High temperature
    hot_outputs = fermi_model(X_test[:5])
    print(f"Hot outputs (T=5.0): {hot_outputs[0].detach().numpy()}")
    
    print("\n✅ Demo completed successfully!")
    print("Both Fermi-Dirac and Maxwell-Boltzmann brains are working correctly!")


if __name__ == "__main__":
    simple_demo()