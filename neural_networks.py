"""
Neural Network implementations that require minimal or no training.

This module contains implementations of various neural network architectures
that can work effectively without traditional backpropagation training,
including random feature networks, reservoir computing, and other approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple


class RandomFeatureNetwork(nn.Module):
    """
    A neural network with random fixed weights that doesn't require training.
    Only the output layer is trainable (if needed).
    
    This is based on the concept that random features can be surprisingly effective
    for many tasks when using sufficient width.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 activation: str = 'relu', bias: bool = True):
        super(RandomFeatureNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Random hidden layer (fixed, not trainable)
        self.hidden_layer = nn.Linear(input_size, hidden_size, bias=bias)
        
        # Freeze the hidden layer weights
        for param in self.hidden_layer.parameters():
            param.requires_grad = False
            
        # Initialize hidden weights with Xavier/Glorot initialization
        with torch.no_grad():
            nn.init.xavier_uniform_(self.hidden_layer.weight)
            if bias:
                nn.init.zeros_(self.hidden_layer.bias)
        
        # Only the output layer is trainable
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Set activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Random feature transformation (non-trainable)
        hidden = self.activation(self.hidden_layer(x))
        
        # Linear output (trainable)
        output = self.output_layer(hidden)
        
        return output
    
    def get_random_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get the random feature representation."""
        return self.activation(self.hidden_layer(x))


class EchoStateNetwork(nn.Module):
    """
    Echo State Network (ESN) - A type of reservoir computing network.
    
    The reservoir (hidden layer) has fixed random weights and only the
    output weights are trained. The key is to have a reservoir with
    the echo state property.
    """
    
    def __init__(self, input_size: int, reservoir_size: int, output_size: int,
                 spectral_radius: float = 0.95, sparsity: float = 0.1,
                 input_scaling: float = 1.0, bias_scaling: float = 1.0):
        super(EchoStateNetwork, self).__init__()
        
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        
        # Input weights (fixed, random)
        self.W_in = nn.Parameter(torch.randn(reservoir_size, input_size) * input_scaling,
                                requires_grad=False)
        
        # Reservoir weights (fixed, random, sparse)
        W_res = torch.randn(reservoir_size, reservoir_size)
        
        # Make sparse
        mask = torch.rand(reservoir_size, reservoir_size) < sparsity
        W_res = W_res * mask.float()
        
        # Scale to desired spectral radius
        eigenvalues = torch.linalg.eigvals(W_res)
        current_spectral_radius = torch.max(torch.abs(eigenvalues)).real
        W_res = W_res * (spectral_radius / current_spectral_radius)
        
        self.W_res = nn.Parameter(W_res, requires_grad=False)
        
        # Bias (fixed, random)
        self.bias = nn.Parameter(torch.randn(reservoir_size) * bias_scaling,
                                requires_grad=False)
        
        # Output weights (trainable)
        self.W_out = nn.Linear(reservoir_size, output_size)
        
        # Hidden state
        self.hidden_state = None
    
    def reset_state(self, batch_size: int = 1):
        """Reset the hidden state."""
        self.hidden_state = torch.zeros(batch_size, self.reservoir_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ESN."""
        batch_size = x.size(0)
        
        if self.hidden_state is None or self.hidden_state.size(0) != batch_size:
            self.reset_state(batch_size)
        
        # Update reservoir state
        # h(t) = tanh(W_in * x(t) + W_res * h(t-1) + bias)
        input_activation = torch.matmul(x, self.W_in.t())
        reservoir_activation = torch.matmul(self.hidden_state, self.W_res.t())
        
        self.hidden_state = torch.tanh(input_activation + reservoir_activation + self.bias)
        
        # Compute output
        output = self.W_out(self.hidden_state)
        
        return output


class ExtremeLearnlingMachine(nn.Module):
    """
    Extreme Learning Machine (ELM) - Single hidden layer with random weights.
    
    Only the output weights are computed analytically or trained.
    The hidden layer weights and biases are randomly assigned and fixed.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 activation: str = 'sigmoid'):
        super(ExtremeLearnlingMachine, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Random input weights and biases (fixed)
        self.input_weights = nn.Parameter(
            torch.randn(input_size, hidden_size), requires_grad=False
        )
        self.biases = nn.Parameter(
            torch.randn(hidden_size), requires_grad=False
        )
        
        # Output weights (trainable or computed analytically)
        self.output_weights = nn.Parameter(
            torch.randn(hidden_size, output_size), requires_grad=True
        )
        
        # Activation function
        if activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ELM."""
        # Hidden layer activation
        hidden = self.activation(torch.matmul(x, self.input_weights) + self.biases)
        
        # Output layer
        output = torch.matmul(hidden, self.output_weights)
        
        return output
    
    def fit_analytical(self, x: torch.Tensor, y: torch.Tensor, reg_lambda: float = 1e-6):
        """
        Analytically compute output weights using Moore-Penrose pseudoinverse.
        
        Args:
            x: Input data [batch_size, input_size]
            y: Target data [batch_size, output_size]
            reg_lambda: Regularization parameter
        """
        with torch.no_grad():
            # Compute hidden layer activations
            hidden = self.activation(torch.matmul(x, self.input_weights) + self.biases)
            
            # Compute output weights using regularized least squares
            # W_out = (H^T H + λI)^(-1) H^T Y
            H = hidden
            HTH = torch.matmul(H.t(), H)
            HTH_reg = HTH + reg_lambda * torch.eye(self.hidden_size)
            HTY = torch.matmul(H.t(), y)
            
            self.output_weights.data = torch.linalg.solve(HTH_reg, HTY)


class LiquidStateMachine(nn.Module):
    """
    Liquid State Machine (LSM) - A spiking neural network reservoir.
    
    This is a simplified implementation that uses continuous values
    instead of actual spikes for demonstration purposes.
    """
    
    def __init__(self, input_size: int, liquid_size: int, output_size: int,
                 time_constant: float = 10.0, threshold: float = 1.0):
        super(LiquidStateMachine, self).__init__()
        
        self.input_size = input_size
        self.liquid_size = liquid_size
        self.output_size = output_size
        self.time_constant = time_constant
        self.threshold = threshold
        
        # Input connections (fixed, random, sparse)
        input_conn = torch.randn(liquid_size, input_size) * 0.5
        input_mask = torch.rand(liquid_size, input_size) < 0.3
        self.W_in = nn.Parameter(input_conn * input_mask.float(), requires_grad=False)
        
        # Liquid connections (fixed, random, sparse)
        liquid_conn = torch.randn(liquid_size, liquid_size) * 0.3
        liquid_mask = torch.rand(liquid_size, liquid_size) < 0.1
        # Remove self-connections
        liquid_mask.fill_diagonal_(False)
        self.W_liquid = nn.Parameter(liquid_conn * liquid_mask.float(), requires_grad=False)
        
        # Output weights (trainable)
        self.readout = nn.Linear(liquid_size, output_size)
        
        # Liquid state variables
        self.membrane_potential = None
        self.spike_history = None
    
    def reset_state(self, batch_size: int = 1):
        """Reset the liquid state."""
        self.membrane_potential = torch.zeros(batch_size, self.liquid_size)
        self.spike_history = torch.zeros(batch_size, self.liquid_size)
    
    def forward(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Forward pass through the LSM."""
        batch_size = x.size(0)
        
        if self.membrane_potential is None or self.membrane_potential.size(0) != batch_size:
            self.reset_state(batch_size)
        
        # Input current
        I_input = torch.matmul(x, self.W_in.t())
        
        # Recurrent current from liquid
        I_recurrent = torch.matmul(self.spike_history, self.W_liquid.t())
        
        # Update membrane potential (leaky integrate)
        decay = torch.exp(torch.tensor(-dt / self.time_constant))
        self.membrane_potential = (decay * self.membrane_potential + 
                                 dt * (I_input + I_recurrent))
        
        # Generate spikes (simplified)
        spikes = (self.membrane_potential > self.threshold).float()
        
        # Reset spiked neurons
        self.membrane_potential = self.membrane_potential * (1 - spikes)
        
        # Update spike history
        self.spike_history = spikes
        
        # Readout
        output = self.readout(self.spike_history)
        
        return output


def create_random_dataset(n_samples: int = 1000, input_dim: int = 10, 
                         output_dim: int = 1, noise_level: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a random dataset for testing the networks.
    
    Args:
        n_samples: Number of samples
        input_dim: Input dimensionality
        output_dim: Output dimensionality
        noise_level: Noise level in the targets
    
    Returns:
        Tuple of (inputs, targets)
    """
    X = torch.randn(n_samples, input_dim)
    
    # Create a non-linear target function
    W_true = torch.randn(input_dim, output_dim)
    y = torch.matmul(X, W_true) + torch.sin(torch.sum(X, dim=1, keepdim=True))
    y += noise_level * torch.randn_like(y)
    
    return X, y


if __name__ == "__main__":
    # Example usage and demonstrations
    print("Neural Networks that don't need training!")
    print("=" * 50)
    
    # Create sample data
    X_train, y_train = create_random_dataset(500, 20, 3)
    X_test, y_test = create_random_dataset(100, 20, 3)
    
    print(f"Dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Input dim: {X_train.shape[1]}, Output dim: {y_train.shape[1]}")
    print()
    
    # 1. Random Feature Network
    print("1. Random Feature Network")
    rfn = RandomFeatureNetwork(20, 100, 3)
    print(f"   Total parameters: {sum(p.numel() for p in rfn.parameters())}")
    print(f"   Trainable parameters: {sum(p.numel() for p in rfn.parameters() if p.requires_grad)}")
    
    # Test forward pass
    with torch.no_grad():
        output = rfn(X_test)
        print(f"   Output shape: {output.shape}")
    print()
    
    # 2. Echo State Network
    print("2. Echo State Network")
    esn = EchoStateNetwork(20, 50, 3)
    print(f"   Total parameters: {sum(p.numel() for p in esn.parameters())}")
    print(f"   Trainable parameters: {sum(p.numel() for p in esn.parameters() if p.requires_grad)}")
    
    # Test forward pass
    esn.reset_state(X_test.size(0))
    with torch.no_grad():
        output = esn(X_test)
        print(f"   Output shape: {output.shape}")
    print()
    
    # 3. Extreme Learning Machine
    print("3. Extreme Learning Machine")
    elm = ExtremeLearnlingMachine(20, 80, 3)
    print(f"   Total parameters: {sum(p.numel() for p in elm.parameters())}")
    print(f"   Trainable parameters: {sum(p.numel() for p in elm.parameters() if p.requires_grad)}")
    
    # Analytical training
    elm.fit_analytical(X_train, y_train)
    
    # Test forward pass
    with torch.no_grad():
        output = elm(X_test)
        mse = F.mse_loss(output, y_test)
        print(f"   Output shape: {output.shape}")
        print(f"   Test MSE: {mse.item():.6f}")
    print()
    
    # 4. Liquid State Machine
    print("4. Liquid State Machine")
    lsm = LiquidStateMachine(20, 60, 3)
    print(f"   Total parameters: {sum(p.numel() for p in lsm.parameters())}")
    print(f"   Trainable parameters: {sum(p.numel() for p in lsm.parameters() if p.requires_grad)}")
    
    # Test forward pass
    lsm.reset_state(X_test.size(0))
    with torch.no_grad():
        output = lsm(X_test)
        print(f"   Output shape: {output.shape}")
    print()
    
    print("All networks created successfully!")
    print("These networks can work without traditional backpropagation training!")