"""
Maxwell-Boltzmann Brain Neural Network

This module implements a neural network based on Maxwell-Boltzmann statistics,
which describes the distribution of classical particles over energy states
in statistical mechanics.

The Maxwell-Boltzmann distribution is: f(E) = exp(-E/kT)
where E is energy, k is Boltzmann constant, T is temperature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MaxwellBoltzmannActivation(nn.Module):
    """
    Activation function based on Maxwell-Boltzmann distribution.
    
    The Maxwell-Boltzmann function provides an exponential decay activation
    that represents the probability distribution of classical particles.
    """
    
    def __init__(self, temperature=1.0, energy_shift=0.0):
        super(MaxwellBoltzmannActivation, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))
        self.energy_shift = nn.Parameter(torch.tensor(energy_shift, dtype=torch.float32))
    
    def forward(self, x):
        """
        Apply Maxwell-Boltzmann activation: f(x) = exp(-(x - E₀)/T)
        """
        # Shift energy levels and apply temperature scaling
        energy = (x - self.energy_shift) / (self.temperature + 1e-8)
        # Clamp more conservatively to prevent overflow
        energy = torch.clamp(energy, -10, 10)  # Reduced from -50, 50 to -10, 10
        return torch.exp(-energy)


class MaxwellBoltzmannLayer(nn.Module):
    """
    A neural network layer that incorporates Maxwell-Boltzmann statistics.
    
    This layer uses Maxwell-Boltzmann principles for both weight initialization
    and activation functions.
    """
    
    def __init__(self, in_features, out_features, temperature=1.0, use_bias=True):
        super(MaxwellBoltzmannLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.temperature = temperature
        
        # Initialize weights using Maxwell-Boltzmann inspired initialization
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if use_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.activation = MaxwellBoltzmannActivation(temperature=temperature)
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Initialize parameters using Maxwell-Boltzmann inspired method.
        
        Weights are initialized to follow an exponential distribution
        that reflects classical thermal equilibrium.
        """
        # Initialize weights following Maxwell-Boltzmann energy distribution
        fan_in = self.in_features
        
        with torch.no_grad():
            # Generate weights from exponential distribution (Maxwell-Boltzmann like)
            # Use a more conservative initialization to prevent overflow
            weights = torch.rand_like(self.weight)
            # Convert uniform random to exponential using inverse transform sampling
            weights = -torch.log(weights + 1e-8)  # Add small epsilon to prevent log(0)
            
            # Scale more conservatively to prevent numerical issues
            scale = (self.temperature * 0.1) / math.sqrt(fan_in)  # Reduce by factor of 10
            weights = weights * scale
            
            # Randomly assign positive and negative values
            signs = torch.randint(0, 2, self.weight.shape, dtype=torch.float32) * 2 - 1
            self.weight.data = weights * signs
            
        if self.bias is not None:
            with torch.no_grad():
                bound = 1 / math.sqrt(fan_in)
                self.bias.uniform_(-bound, bound)
    
    def forward(self, x):
        """Forward pass through the Maxwell-Boltzmann layer."""
        linear_output = F.linear(x, self.weight, self.bias)
        return self.activation(linear_output)


class EnergyNormalization(nn.Module):
    """
    Normalization layer that maintains energy conservation principles.
    
    This ensures that the total "energy" of activations is conserved,
    similar to how energy is conserved in classical systems.
    """
    
    def __init__(self, num_features, eps=1e-5):
        super(EnergyNormalization, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x):
        """
        Normalize based on energy conservation.
        """
        # Compute "energy" as sum of squares
        energy = torch.sum(x * x, dim=-1, keepdim=True)
        mean_energy = torch.mean(energy)
        
        # Normalize to maintain average energy
        normalized = x / torch.sqrt(energy / mean_energy + self.eps)
        
        # Apply learned scaling and shifting
        return self.weight * normalized + self.bias


class MaxwellBoltzmannBrain(nn.Module):
    """
    Complete neural network based on Maxwell-Boltzmann statistics.
    
    This network uses Maxwell-Boltzmann layers and incorporates classical
    statistical mechanics principles into the neural network architecture.
    """
    
    def __init__(self, input_size, hidden_sizes, output_size, temperature=1.0, 
                 use_energy_norm=True, dropout_rate=0.1):
        super(MaxwellBoltzmannBrain, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes if isinstance(hidden_sizes, list) else [hidden_sizes]
        self.output_size = output_size
        self.temperature = temperature
        self.use_energy_norm = use_energy_norm
        
        # Build the network layers
        layers = []
        prev_size = input_size
        
        # Hidden layers with Maxwell-Boltzmann activation
        for i, hidden_size in enumerate(self.hidden_sizes):
            layers.append(MaxwellBoltzmannLayer(prev_size, hidden_size, temperature))
            
            # Add energy normalization if requested
            if use_energy_norm:
                layers.append(EnergyNormalization(hidden_size))
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer (linear for flexibility in loss functions)
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Store thermodynamic properties
        self.thermal_energy = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.partition_function = nn.Parameter(torch.tensor(1.0), requires_grad=False)
    
    def forward(self, x):
        """Forward pass through the Maxwell-Boltzmann brain."""
        return self.network(x)
    
    def compute_thermal_energy(self):
        """
        Compute the average thermal energy of the network.
        
        This provides insight into the "temperature" state of the network.
        """
        total_energy = 0.0
        num_params = 0
        
        for module in self.modules():
            if isinstance(module, MaxwellBoltzmannLayer):
                with torch.no_grad():
                    # Compute energy as sum of squared weights
                    energy = torch.sum(module.weight ** 2)
                    total_energy += energy.item()
                    num_params += module.weight.numel()
        
        if num_params > 0:
            avg_energy = total_energy / num_params
            self.thermal_energy.data = torch.tensor(avg_energy)
        
        return self.thermal_energy.item()
    
    def compute_partition_function(self):
        """
        Compute the partition function of the network.
        
        This represents the sum over all possible states weighted by their
        Boltzmann factors.
        """
        total_z = 0.0
        num_layers = 0
        
        for module in self.modules():
            if isinstance(module, MaxwellBoltzmannActivation):
                with torch.no_grad():
                    # Estimate partition function based on current parameters
                    energy_scale = module.temperature + 1e-8
                    # Approximate Z as the effective number of accessible states
                    z = energy_scale * math.sqrt(2 * math.pi)
                    total_z += z
                    num_layers += 1
        
        if num_layers > 0:
            self.partition_function.data = torch.tensor(total_z / num_layers)
        
        return self.partition_function.item()
    
    def adjust_temperature(self, new_temperature):
        """
        Adjust the temperature of all Maxwell-Boltzmann layers.
        
        This allows for "thermal annealing" of the network during training.
        """
        self.temperature = new_temperature
        for module in self.modules():
            if isinstance(module, (MaxwellBoltzmannActivation, MaxwellBoltzmannLayer)):
                if hasattr(module, 'temperature') and hasattr(module.temperature, 'data'):
                    module.temperature.data = torch.tensor(new_temperature, dtype=torch.float32)
                elif isinstance(module, MaxwellBoltzmannLayer):
                    module.temperature = new_temperature
    
    def thermal_anneal(self, current_epoch, total_epochs, initial_temp=2.0, final_temp=0.1):
        """
        Perform thermal annealing during training.
        
        Temperature decreases according to an annealing schedule,
        simulating cooling in classical systems.
        """
        progress = current_epoch / total_epochs
        # Exponential annealing
        current_temp = initial_temp * (final_temp / initial_temp) ** progress
        self.adjust_temperature(current_temp)
        return current_temp
    
    def get_average_energy_shift(self):
        """
        Get the average energy shift across all layers.
        """
        shifts = []
        for module in self.modules():
            if isinstance(module, MaxwellBoltzmannActivation):
                shifts.append(module.energy_shift.item())
        
        return sum(shifts) / len(shifts) if shifts else 0.0


# Specialized architectures
class ClassicalGasNet(MaxwellBoltzmannBrain):
    """
    Neural network that mimics the behavior of classical gas particles.
    
    Uses higher temperatures and energy normalization to simulate
    particle interactions in a classical gas.
    """
    
    def __init__(self, input_size, hidden_sizes, output_size, initial_temperature=2.0):
        super(ClassicalGasNet, self).__init__(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            temperature=initial_temperature,
            use_energy_norm=True,
            dropout_rate=0.05  # Lower dropout to maintain "particle interactions"
        )


# Example usage and utility functions
def create_maxwell_boltzmann_classifier(input_size, num_classes, hidden_sizes=[64, 32], temperature=1.0):
    """
    Create a Maxwell-Boltzmann brain for classification tasks.
    """
    return MaxwellBoltzmannBrain(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=num_classes,
        temperature=temperature
    )


def create_maxwell_boltzmann_regressor(input_size, output_size=1, hidden_sizes=[64, 32], temperature=1.0):
    """
    Create a Maxwell-Boltzmann brain for regression tasks.
    """
    return MaxwellBoltzmannBrain(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        temperature=temperature
    )


def create_classical_gas_network(input_size, hidden_sizes, output_size, temperature=2.0):
    """
    Create a classical gas network that simulates particle dynamics.
    """
    return ClassicalGasNet(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        initial_temperature=temperature
    )