"""
Example usage of Maxwell-Boltzmann Brain Neural Network

This script demonstrates how to use the Maxwell-Boltzmann Brain for classification
and regression tasks, showcasing classical statistical mechanics-inspired neural network behavior.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from maxwell_boltzmann_brain import (
    MaxwellBoltzmannBrain, 
    create_maxwell_boltzmann_classifier, 
    create_maxwell_boltzmann_regressor,
    create_classical_gas_network
)


def maxwell_boltzmann_classification_example():
    """
    Example of using Maxwell-Boltzmann Brain for binary classification.
    """
    print("=== Maxwell-Boltzmann Brain Classification Example ===")
    
    # Generate synthetic classification dataset
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
        n_classes=2, random_state=42
    )
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    # Create Maxwell-Boltzmann brain
    model = create_maxwell_boltzmann_classifier(
        input_size=20,
        num_classes=2,
        hidden_sizes=[64, 32, 16],
        temperature=2.0
    )
    
    print(f"Model architecture: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop with thermal annealing
    epochs = 100
    train_losses = []
    test_accuracies = []
    thermal_energies = []
    temperatures = []
    
    for epoch in range(epochs):
        # Thermal annealing
        current_temp = model.thermal_anneal(epoch, epochs, initial_temp=2.0, final_temp=0.1)
        temperatures.append(current_temp)
        
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs.data, 1)
            accuracy = (predicted == y_test).float().mean().item()
            test_accuracies.append(accuracy)
            
            # Compute thermodynamic properties
            thermal_energy = model.compute_thermal_energy()
            thermal_energies.append(thermal_energy)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, "
                  f"Test Accuracy: {accuracy:.4f}, Thermal Energy: {thermal_energy:.4f}, "
                  f"Temperature: {current_temp:.3f}")
    
    # Final evaluation
    final_accuracy = test_accuracies[-1]
    final_energy = thermal_energies[-1]
    partition_function = model.compute_partition_function()
    
    print(f"\nFinal Results:")
    print(f"Test Accuracy: {final_accuracy:.4f}")
    print(f"Thermal Energy: {final_energy:.4f}")
    print(f"Partition Function: {partition_function:.4f}")
    
    # Plot training progress
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 4, 2)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 4, 3)
    plt.plot(thermal_energies)
    plt.title('Thermal Energy')
    plt.xlabel('Epoch')
    plt.ylabel('Energy')
    
    plt.subplot(1, 4, 4)
    plt.plot(temperatures)
    plt.title('Temperature (Annealing)')
    plt.xlabel('Epoch')
    plt.ylabel('Temperature')
    
    plt.tight_layout()
    plt.savefig('/tmp/maxwell_boltzmann_classification.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return model, final_accuracy


def classical_gas_network_example():
    """
    Example of using Classical Gas Network for multi-class classification.
    """
    print("\n=== Classical Gas Network Example ===")
    
    # Generate synthetic multi-class dataset
    X, y = make_classification(
        n_samples=1500, n_features=15, n_informative=12, n_redundant=3,
        n_classes=3, n_clusters_per_class=1, random_state=42
    )
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    # Create Classical Gas Network
    model = create_classical_gas_network(
        input_size=15,
        hidden_sizes=[128, 64, 32],
        output_size=3,
        temperature=3.0
    )
    
    print(f"Model architecture: {model}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 120
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs.data, 1)
            accuracy = (predicted == y_test).float().mean().item()
            test_accuracies.append(accuracy)
        
        if (epoch + 1) % 30 == 0:
            thermal_energy = model.compute_thermal_energy()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, "
                  f"Test Accuracy: {accuracy:.4f}, Thermal Energy: {thermal_energy:.4f}")
    
    final_accuracy = test_accuracies[-1]
    print(f"\nClassical Gas Network Final Accuracy: {final_accuracy:.4f}")
    
    return model, final_accuracy


def maxwell_boltzmann_regression_example():
    """
    Example of using Maxwell-Boltzmann Brain for regression.
    """
    print("\n=== Maxwell-Boltzmann Brain Regression Example ===")
    
    # Generate synthetic regression dataset
    X, y = make_regression(
        n_samples=1000, n_features=10, noise=0.1, random_state=42
    )
    
    # Normalize features and targets
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    
    # Create Maxwell-Boltzmann brain
    model = create_maxwell_boltzmann_regressor(
        input_size=10,
        output_size=1,
        hidden_sizes=[64, 32],
        temperature=1.5
    )
    
    print(f"Model architecture: {model}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop with thermal annealing
    epochs = 150
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # Thermal annealing every 25 epochs
        if epoch % 25 == 0 and epoch > 0:
            current_temp = model.thermal_anneal(epoch, epochs, initial_temp=1.5, final_temp=0.2)
        
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test).squeeze()
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())
        
        if (epoch + 1) % 30 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, "
                  f"Test Loss: {test_loss.item():.4f}")
    
    final_test_loss = test_losses[-1]
    print(f"\nFinal Test MSE: {final_test_loss:.4f}")
    
    return model, final_test_loss


def temperature_comparison_analysis():
    """
    Compare Maxwell-Boltzmann brain performance at different temperatures.
    """
    print("\n=== Temperature Comparison Analysis ===")
    
    # Generate a simple dataset
    X, y = make_classification(
        n_samples=800, n_features=12, n_informative=10, n_classes=2, random_state=42
    )
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    results = []
    
    for temp in temperatures:
        print(f"Testing temperature: {temp}")
        
        model = create_maxwell_boltzmann_classifier(
            input_size=12, num_classes=2, hidden_sizes=[48, 24], temperature=temp
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Quick training
        for epoch in range(60):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs.data, 1)
            accuracy = (predicted == y_test).float().mean().item()
            thermal_energy = model.compute_thermal_energy()
            partition_function = model.compute_partition_function()
        
        results.append({
            'temperature': temp,
            'accuracy': accuracy,
            'thermal_energy': thermal_energy,
            'partition_function': partition_function
        })
        
        print(f"Accuracy: {accuracy:.4f}, Thermal Energy: {thermal_energy:.4f}")
    
    # Plot results
    temps = [r['temperature'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    energies = [r['thermal_energy'] for r in results]
    partitions = [r['partition_function'] for r in results]
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.semilogx(temps, accuracies, 'bo-')
    plt.xlabel('Temperature')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy vs Temperature')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.semilogx(temps, energies, 'ro-')
    plt.xlabel('Temperature')
    plt.ylabel('Thermal Energy')
    plt.title('Thermal Energy vs Temperature')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.semilogx(temps, partitions, 'go-')
    plt.xlabel('Temperature')
    plt.ylabel('Partition Function')
    plt.title('Partition Function vs Temperature')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/tmp/maxwell_boltzmann_temperature_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results


def energy_conservation_demo():
    """
    Demonstrate energy conservation properties of the network.
    """
    print("\n=== Energy Conservation Demonstration ===")
    
    # Create a simple network with energy normalization
    model = MaxwellBoltzmannBrain(
        input_size=5,
        hidden_sizes=[10, 10],
        output_size=1,
        temperature=1.0,
        use_energy_norm=True
    )
    
    # Generate random inputs
    batch_size = 100
    inputs = torch.randn(batch_size, 5)
    
    # Track energy through layers
    model.eval()
    with torch.no_grad():
        x = inputs
        layer_energies = []
        
        for i, layer in enumerate(model.network):
            if isinstance(layer, nn.Linear) or hasattr(layer, 'forward'):
                x = layer(x)
                # Compute total energy (sum of squares)
                energy = torch.sum(x ** 2, dim=1).mean().item()
                layer_energies.append(energy)
                print(f"Layer {i} average energy: {energy:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(layer_energies)), layer_energies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Layer Index')
    plt.ylabel('Average Energy')
    plt.title('Energy Flow Through Maxwell-Boltzmann Network')
    plt.grid(True)
    plt.savefig('/tmp/energy_conservation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return layer_energies


if __name__ == "__main__":
    # Run all examples
    try:
        # Classification example
        model_cls, acc_cls = maxwell_boltzmann_classification_example()
        
        # Classical gas network example
        gas_model, acc_gas = classical_gas_network_example()
        
        # Regression example
        model_reg, mse = maxwell_boltzmann_regression_example()
        
        # Temperature analysis
        temp_results = temperature_comparison_analysis()
        
        # Energy conservation demo
        energies = energy_conservation_demo()
        
        print("\n=== Summary ===")
        print(f"Classification accuracy (MB Brain): {acc_cls:.4f}")
        print(f"Classification accuracy (Gas Network): {acc_gas:.4f}")
        print(f"Regression MSE: {mse:.4f}")
        print("Temperature analysis completed successfully!")
        print("Energy conservation demonstration completed!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()