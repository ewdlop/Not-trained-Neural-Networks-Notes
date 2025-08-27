"""
Examples demonstrating the different neural network architectures
that require minimal or no training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from neural_networks import (
    RandomFeatureNetwork, EchoStateNetwork, 
    ExtremeLearnlingMachine, LiquidStateMachine,
    create_random_dataset
)


def compare_networks():
    """Compare different networks on a regression task."""
    print("Comparing Neural Networks that don't need training")
    print("=" * 60)
    
    # Create datasets
    X_train, y_train = create_random_dataset(1000, 15, 1, noise_level=0.1)
    X_test, y_test = create_random_dataset(200, 15, 1, noise_level=0.1)
    
    # Networks to compare
    networks = {
        'Random Features': RandomFeatureNetwork(15, 200, 1),
        'Echo State': EchoStateNetwork(15, 100, 1),
        'Extreme Learning': ExtremeLearnlingMachine(15, 150, 1),
        'Liquid State': LiquidStateMachine(15, 80, 1)
    }
    
    results = {}
    
    for name, network in networks.items():
        print(f"\n{name} Network:")
        print(f"  Total params: {sum(p.numel() for p in network.parameters())}")
        print(f"  Trainable params: {sum(p.numel() for p in network.parameters() if p.requires_grad)}")
        
        if name == 'Extreme Learning':
            # ELM can be trained analytically
            network.fit_analytical(X_train, y_train)
            
        elif name in ['Random Features', 'Echo State', 'Liquid State']:
            # Train only the output layer for these networks
            if name == 'Echo State':
                network.reset_state(X_train.size(0))
            elif name == 'Liquid State':
                network.reset_state(X_train.size(0))
                
            optimizer = optim.Adam([p for p in network.parameters() if p.requires_grad], lr=0.01)
            criterion = nn.MSELoss()
            
            # Quick training (only output layer)
            for epoch in range(100):
                optimizer.zero_grad()
                
                if name == 'Echo State':
                    network.reset_state(X_train.size(0))
                elif name == 'Liquid State':
                    network.reset_state(X_train.size(0))
                
                outputs = network(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
                
                if epoch % 25 == 0:
                    print(f"    Epoch {epoch}, Loss: {loss.item():.6f}")
        
        # Evaluate
        network.eval()
        with torch.no_grad():
            if name == 'Echo State':
                network.reset_state(X_test.size(0))
            elif name == 'Liquid State':
                network.reset_state(X_test.size(0))
                
            test_outputs = network(X_test)
            test_loss = nn.MSELoss()(test_outputs, y_test)
            
        print(f"  Test MSE: {test_loss.item():.6f}")
        results[name] = test_loss.item()
    
    # Print summary
    print(f"\n{'Network':<20} {'Test MSE':<15} {'Rank'}")
    print("-" * 40)
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for i, (name, mse) in enumerate(sorted_results, 1):
        print(f"{name:<20} {mse:<15.6f} {i}")


def demonstration_with_visualization():
    """Create a visual demonstration of the networks."""
    print("\nVisual Demonstration")
    print("=" * 30)
    
    # Create a simple 1D regression problem
    x = torch.linspace(-3, 3, 300).unsqueeze(1)
    y_true = torch.sin(2*x) + 0.5*torch.sin(5*x) + 0.1*torch.randn_like(x)
    
    # Train subset
    train_indices = torch.randperm(300)[:200]
    x_train = x[train_indices]
    y_train = y_true[train_indices]
    
    # Create networks
    networks = {
        'Random Features (ReLU)': RandomFeatureNetwork(1, 100, 1, activation='relu'),
        'Random Features (Tanh)': RandomFeatureNetwork(1, 100, 1, activation='tanh'),
        'ELM (Sigmoid)': ExtremeLearnlingMachine(1, 80, 1, activation='sigmoid'),
    }
    
    plt.figure(figsize=(15, 10))
    
    # Plot true function
    plt.subplot(2, 2, 1)
    plt.scatter(x_train.numpy(), y_train.numpy(), alpha=0.6, s=20, label='Training Data')
    plt.plot(x.numpy(), y_true.numpy(), 'r-', alpha=0.8, linewidth=2, label='True Function')
    plt.title('Original Function and Training Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_idx = 2
    for name, network in networks.items():
        if 'ELM' in name:
            # Analytical solution for ELM
            network.fit_analytical(x_train, y_train)
        else:
            # Train output layer only
            optimizer = optim.Adam([p for p in network.parameters() if p.requires_grad], lr=0.01)
            criterion = nn.MSELoss()
            
            for epoch in range(200):
                optimizer.zero_grad()
                outputs = network(x_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
        
        # Generate predictions
        network.eval()
        with torch.no_grad():
            y_pred = network(x)
        
        plt.subplot(2, 2, plot_idx)
        plt.scatter(x_train.numpy(), y_train.numpy(), alpha=0.6, s=20, label='Training Data')
        plt.plot(x.numpy(), y_true.numpy(), 'r-', alpha=0.8, linewidth=2, label='True Function')
        plt.plot(x.numpy(), y_pred.numpy(), 'b-', alpha=0.8, linewidth=2, label='Prediction')
        plt.title(f'{name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calculate MSE
        mse = nn.MSELoss()(y_pred, y_true)
        plt.text(0.02, 0.98, f'MSE: {mse.item():.4f}', transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('network_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved visualization as 'network_comparison.png'")
    plt.close()


def showcase_reservoir_computing():
    """Demonstrate reservoir computing with time series."""
    print("\nReservoir Computing Time Series Demo")
    print("=" * 40)
    
    # Generate a time series (sine wave with noise)
    time_points = torch.linspace(0, 10*np.pi, 1000)
    signal = torch.sin(time_points) + 0.3*torch.sin(3*time_points) + 0.1*torch.randn_like(time_points)
    
    # Prepare sequences for prediction
    seq_length = 10
    X_sequences = []
    y_sequences = []
    
    for i in range(len(signal) - seq_length):
        X_sequences.append(signal[i:i+seq_length])
        y_sequences.append(signal[i+seq_length])
    
    X = torch.stack(X_sequences).unsqueeze(-1)  # [batch, seq, 1]
    y = torch.stack(y_sequences).unsqueeze(-1)  # [batch, 1]
    
    # Split data
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Time series length: {len(signal)}")
    print(f"Sequences: {len(X)} (train: {len(X_train)}, test: {len(X_test)})")
    
    # Create ESN for sequence prediction
    esn = EchoStateNetwork(1, 50, 1, spectral_radius=0.9)
    
    # Train (only output layer)
    optimizer = optim.Adam(esn.W_out.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("\nTraining ESN...")
    for epoch in range(100):
        total_loss = 0
        esn.reset_state(1)
        
        for i in range(len(X_train)):
            optimizer.zero_grad()
            
            # Feed sequence one step at a time
            for t in range(seq_length):
                if t == seq_length - 1:
                    # Only compute loss on final prediction
                    output = esn(X_train[i:i+1, t:t+1])
                    loss = criterion(output, y_train[i:i+1])
                    loss.backward()
                else:
                    _ = esn(X_train[i:i+1, t:t+1])
            
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}, Loss: {total_loss/len(X_train):.6f}")
    
    # Test
    esn.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(len(X_test)):
            esn.reset_state(1)
            for t in range(seq_length):
                if t == seq_length - 1:
                    pred = esn(X_test[i:i+1, t:t+1])
                    predictions.append(pred.item())
                else:
                    _ = esn(X_test[i:i+1, t:t+1])
    
    test_mse = nn.MSELoss()(torch.tensor(predictions), y_test.squeeze())
    print(f"\nTest MSE: {test_mse.item():.6f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    test_indices = np.arange(train_size, train_size + len(predictions))
    
    plt.plot(time_points.numpy(), signal.numpy(), 'b-', alpha=0.7, label='True Signal')
    plt.plot(time_points[test_indices].numpy(), predictions, 'r-', alpha=0.8, label='ESN Predictions')
    plt.axvline(x=time_points[train_size].item(), color='g', linestyle='--', alpha=0.7, label='Train/Test Split')
    plt.title('Echo State Network Time Series Prediction')
    plt.xlabel('Time')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('time_series_prediction.png', dpi=150, bbox_inches='tight')
    print("Saved time series plot as 'time_series_prediction.png'")
    plt.close()


if __name__ == "__main__":
    # Run all demonstrations
    compare_networks()
    demonstration_with_visualization()
    showcase_reservoir_computing()
    
    print(f"\n{'='*60}")
    print("All demonstrations completed successfully!")
    print("Check the generated PNG files for visualizations.")