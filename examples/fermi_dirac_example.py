"""
Example usage of Fermi-Dirac Brain Neural Network

This script demonstrates how to use the Fermi-Dirac Brain for classification
and regression tasks, showcasing quantum-inspired neural network behavior.
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

from fermi_dirac_brain import FermiDiracBrain, create_fermi_dirac_classifier, create_fermi_dirac_regressor


def fermi_dirac_classification_example():
    """
    Example of using Fermi-Dirac Brain for binary classification.
    """
    print("=== Fermi-Dirac Brain Classification Example ===")
    
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
    
    # Create Fermi-Dirac brain
    model = create_fermi_dirac_classifier(
        input_size=20,
        num_classes=2,
        hidden_sizes=[64, 32, 16],
        temperature=1.0
    )
    
    print(f"Model architecture: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 100
    train_losses = []
    test_accuracies = []
    quantum_entropies = []
    
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
            
            # Compute quantum entropy
            entropy = model.compute_quantum_entropy()
            quantum_entropies.append(entropy)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, "
                  f"Test Accuracy: {accuracy:.4f}, Quantum Entropy: {entropy:.4f}")
    
    # Final evaluation
    final_accuracy = test_accuracies[-1]
    final_entropy = quantum_entropies[-1]
    fermi_energy = model.get_fermi_energy()
    
    print(f"\nFinal Results:")
    print(f"Test Accuracy: {final_accuracy:.4f}")
    print(f"Quantum Entropy: {final_entropy:.4f}")
    print(f"Fermi Energy: {fermi_energy:.4f}")
    
    # Plot training progress
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 3, 3)
    plt.plot(quantum_entropies)
    plt.title('Quantum Entropy')
    plt.xlabel('Epoch')
    plt.ylabel('Entropy')
    
    plt.tight_layout()
    plt.savefig('/tmp/fermi_dirac_classification.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return model, final_accuracy


def fermi_dirac_regression_example():
    """
    Example of using Fermi-Dirac Brain for regression.
    """
    print("\n=== Fermi-Dirac Brain Regression Example ===")
    
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
    
    # Create Fermi-Dirac brain
    model = create_fermi_dirac_regressor(
        input_size=10,
        output_size=1,
        hidden_sizes=[32, 16],
        temperature=0.5
    )
    
    print(f"Model architecture: {model}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop with temperature annealing
    epochs = 150
    train_losses = []
    test_losses = []
    temperatures = []
    
    for epoch in range(epochs):
        # Temperature annealing
        if epoch % 30 == 0 and epoch > 0:
            new_temp = max(0.1, model.temperature * 0.8)
            model.adjust_temperature(new_temp)
        
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
            
            # Track temperature
            current_temp = model.temperature if hasattr(model, 'temperature') else 1.0
            if hasattr(model.network[0], 'temperature'):
                if hasattr(model.network[0].temperature, 'item'):
                    current_temp = model.network[0].temperature.item()
                else:
                    current_temp = float(model.network[0].temperature)
            temperatures.append(current_temp)
        
        if (epoch + 1) % 30 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, "
                  f"Test Loss: {test_loss.item():.4f}, Temperature: {current_temp:.3f}")
    
    # Final evaluation
    final_test_loss = test_losses[-1]
    print(f"\nFinal Test MSE: {final_test_loss:.4f}")
    
    return model, final_test_loss


def temperature_sensitivity_analysis():
    """
    Analyze how different temperatures affect Fermi-Dirac brain performance.
    """
    print("\n=== Temperature Sensitivity Analysis ===")
    
    # Generate a simple dataset
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=8, n_classes=2, random_state=42
    )
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    results = []
    
    for temp in temperatures:
        print(f"Testing temperature: {temp}")
        
        model = create_fermi_dirac_classifier(
            input_size=10, num_classes=2, hidden_sizes=[32, 16], temperature=temp
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Quick training
        for epoch in range(50):
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
            entropy = model.compute_quantum_entropy()
            fermi_energy = model.get_fermi_energy()
        
        results.append({
            'temperature': temp,
            'accuracy': accuracy,
            'entropy': entropy,
            'fermi_energy': fermi_energy
        })
        
        print(f"Accuracy: {accuracy:.4f}, Entropy: {entropy:.4f}, Fermi Energy: {fermi_energy:.4f}")
    
    # Plot results
    temps = [r['temperature'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    entropies = [r['entropy'] for r in results]
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.semilogx(temps, accuracies, 'bo-')
    plt.xlabel('Temperature')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy vs Temperature')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.semilogx(temps, entropies, 'ro-')
    plt.xlabel('Temperature')
    plt.ylabel('Quantum Entropy')
    plt.title('Quantum Entropy vs Temperature')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/tmp/fermi_dirac_temperature_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results


if __name__ == "__main__":
    # Run all examples
    try:
        # Classification example
        model_cls, acc = fermi_dirac_classification_example()
        
        # Regression example
        model_reg, mse = fermi_dirac_regression_example()
        
        # Temperature analysis
        temp_results = temperature_sensitivity_analysis()
        
        print("\n=== Summary ===")
        print(f"Classification accuracy: {acc:.4f}")
        print(f"Regression MSE: {mse:.4f}")
        print("Temperature analysis completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()