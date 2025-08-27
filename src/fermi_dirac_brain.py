"""
Fermi-Dirac Brain Neural Network

This module implements a neural network based on Fermi-Dirac statistics,
which describes the distribution of fermions (particles with half-integer spin)
over energy states in quantum mechanics.

The Fermi-Dirac distribution is: f(E) = 1 / (exp((E - μ)/kT) + 1)
where E is energy, μ is chemical potential, k is Boltzmann constant, T is temperature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FermiDiracActivation(nn.Module):
    """
    Activation function based on Fermi-Dirac distribution.
    
    The Fermi-Dirac function naturally provides a smooth sigmoid-like activation
    that represents the probability of occupation of energy states by fermions.
    """
    
    def __init__(self, temperature=1.0, chemical_potential=0.0):
        super(FermiDiracActivation, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))
        self.chemical_potential = nn.Parameter(torch.tensor(chemical_potential, dtype=torch.float32))
    
    def forward(self, x):
        """
        Apply Fermi-Dirac activation: f(x) = 1 / (exp((x - μ)/T) + 1)
        """
        # Prevent overflow by clamping the exponent
        exponent = torch.clamp((x - self.chemical_potential) / (self.temperature + 1e-8), -50, 50)
        return 1.0 / (torch.exp(exponent) + 1.0)


class FermiDiracLayer(nn.Module):
    """
    A neural network layer that incorporates Fermi-Dirac statistics.
    
    This layer uses Fermi-Dirac principles for both weight initialization
    and activation functions.
    """
    
    def __init__(self, in_features, out_features, temperature=1.0, use_bias=True):
        super(FermiDiracLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.temperature = temperature
        
        # Initialize weights using Fermi-Dirac inspired initialization
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if use_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.activation = FermiDiracActivation(temperature=temperature)
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Initialize parameters using Fermi-Dirac inspired method.
        
        Weights are initialized to follow a distribution that respects
        the Pauli exclusion principle (no two fermions in the same state).
        """
        # Initialize weights with small random values scaled by 1/sqrt(fan_in)
        # This ensures that the "energy levels" are distributed appropriately
        fan_in = self.in_features
        bound = 1 / math.sqrt(fan_in)
        
        # Use a distribution that naturally limits occupation probability
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            # Apply Fermi-Dirac weighting to initial weights
            energy_levels = torch.linspace(-2, 2, self.weight.numel()).view_as(self.weight)
            fermi_weights = 1.0 / (torch.exp(energy_levels / self.temperature) + 1.0)
            self.weight.mul_(fermi_weights)
            
        if self.bias is not None:
            with torch.no_grad():
                self.bias.uniform_(-bound, bound)
    
    def forward(self, x):
        """Forward pass through the Fermi-Dirac layer."""
        linear_output = F.linear(x, self.weight, self.bias)
        return self.activation(linear_output)


class FermiDiracBrain(nn.Module):
    """
    Complete neural network based on Fermi-Dirac statistics.
    
    This network uses Fermi-Dirac layers and incorporates quantum mechanical
    principles into the neural network architecture.
    """
    
    def __init__(self, input_size, hidden_sizes, output_size, temperature=1.0, dropout_rate=0.1):
        super(FermiDiracBrain, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes if isinstance(hidden_sizes, list) else [hidden_sizes]
        self.output_size = output_size
        self.temperature = temperature
        
        # Build the network layers
        layers = []
        prev_size = input_size
        
        # Hidden layers with Fermi-Dirac activation
        for hidden_size in self.hidden_sizes:
            layers.append(FermiDiracLayer(prev_size, hidden_size, temperature))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer (linear for flexibility in loss functions)
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Store quantum mechanical properties
        self.quantum_entropy = nn.Parameter(torch.tensor(0.0), requires_grad=False)
    
    def forward(self, x):
        """Forward pass through the Fermi-Dirac brain."""
        return self.network(x)
    
    def compute_quantum_entropy(self):
        """
        Compute the quantum entropy of the network based on Fermi-Dirac statistics.
        
        This provides insight into the "quantum state" of the network.
        """
        total_entropy = 0.0
        num_layers = 0
        
        for module in self.modules():
            if isinstance(module, FermiDiracActivation):
                # Get the last activations (would need to be stored during forward pass)
                # For now, compute entropy based on parameters
                with torch.no_grad():
                    p = torch.sigmoid(module.chemical_potential / (module.temperature + 1e-8))
                    p = torch.clamp(p, 1e-8, 1 - 1e-8)  # Avoid log(0)
                    entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
                    total_entropy += entropy
                    num_layers += 1
        
        if num_layers > 0:
            self.quantum_entropy.data = total_entropy / num_layers
        
        return self.quantum_entropy.item()
    
    def adjust_temperature(self, new_temperature):
        """
        Adjust the temperature of all Fermi-Dirac layers.
        
        This allows for "annealing" the network during training.
        """
        for module in self.modules():
            if isinstance(module, (FermiDiracActivation, FermiDiracLayer)):
                if hasattr(module, 'temperature') and hasattr(module.temperature, 'data'):
                    module.temperature.data = torch.tensor(new_temperature, dtype=torch.float32)
                elif isinstance(module, FermiDiracLayer):
                    module.temperature = new_temperature
    
    def get_fermi_energy(self):
        """
        Get the average chemical potential (Fermi energy) of the network.
        """
        potentials = []
        for module in self.modules():
            if isinstance(module, FermiDiracActivation):
                potentials.append(module.chemical_potential.item())
        
        return sum(potentials) / len(potentials) if potentials else 0.0


# Example usage and utility functions
def create_fermi_dirac_classifier(input_size, num_classes, hidden_sizes=[64, 32], temperature=1.0):
    """
    Create a Fermi-Dirac brain for classification tasks.
    """
    return FermiDiracBrain(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=num_classes,
        temperature=temperature
    )


def create_fermi_dirac_regressor(input_size, output_size=1, hidden_sizes=[64, 32], temperature=1.0):
    """
    Create a Fermi-Dirac brain for regression tasks.
    """
    return FermiDiracBrain(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        temperature=temperature
    )