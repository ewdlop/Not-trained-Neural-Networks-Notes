# Statistical Physics-Inspired Neural Networks

This repository implements novel neural network architectures based on fundamental principles from statistical physics and quantum mechanics. The networks incorporate mathematical distributions from quantum and classical statistical mechanics into their design.

## Overview

We implement two main neural network architectures:

1. **Fermi-Dirac Brain** - Based on quantum fermionic statistics
2. **Maxwell-Boltzmann Brain** - Based on classical particle statistics

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Fermi-Dirac Brain](#fermi-dirac-brain)
- [Maxwell-Boltzmann Brain](#maxwell-boltzmann-brain)
- [Examples](#examples)
- [Theory](#theory)
- [API Reference](#api-reference)
- [Contributing](#contributing)

## Installation

```bash
# Clone the repository
git clone https://github.com/ewdlop/Not-trained-Neural-Networks-Notes.git
cd Not-trained-Neural-Networks-Notes

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.7+
- PyTorch 2.0+
- NumPy
- Matplotlib
- Scikit-learn (for examples)

## Quick Start

```python
import sys
sys.path.append('src')

from fermi_dirac_brain import create_fermi_dirac_classifier
from maxwell_boltzmann_brain import create_maxwell_boltzmann_classifier

# Create a Fermi-Dirac neural network
fermi_model = create_fermi_dirac_classifier(
    input_size=10, 
    num_classes=2, 
    hidden_sizes=[64, 32],
    temperature=1.0
)

# Create a Maxwell-Boltzmann neural network
maxwell_model = create_maxwell_boltzmann_classifier(
    input_size=10, 
    num_classes=2, 
    hidden_sizes=[64, 32],
    temperature=2.0
)
```

## Fermi-Dirac Brain

The Fermi-Dirac Brain is inspired by quantum mechanics and fermionic statistics. It incorporates the Fermi-Dirac distribution function into neural network design.

### Key Features

- **Quantum-Inspired Activation**: Uses the Fermi-Dirac distribution as activation function
- **Pauli Exclusion Principle**: Weight initialization respects quantum mechanical principles
- **Temperature Control**: Adjustable temperature parameter for controlling "quantum effects"
- **Quantum Entropy Computation**: Measures the quantum state of the network

### Mathematical Foundation

The Fermi-Dirac distribution describes the probability of occupation of energy states by fermions:

```
f(E) = 1 / (exp((E - μ)/kT) + 1)
```

Where:
- `E` is the energy level (network activations)
- `μ` is the chemical potential (learnable parameter)
- `k` is Boltzmann constant (incorporated into temperature)
- `T` is temperature (controls the "sharpness" of the distribution)

### Usage Example

```python
from fermi_dirac_brain import FermiDiracBrain
import torch

# Create model
model = FermiDiracBrain(
    input_size=20,
    hidden_sizes=[64, 32, 16],
    output_size=2,
    temperature=1.0
)

# Forward pass
x = torch.randn(32, 20)  # batch_size=32, features=20
output = model(x)

# Compute quantum properties
quantum_entropy = model.compute_quantum_entropy()
fermi_energy = model.get_fermi_energy()

# Temperature annealing
model.adjust_temperature(0.5)
```

### Properties

- **Quantum Entropy**: Measures the uncertainty in the quantum state
- **Fermi Energy**: Average chemical potential across layers
- **Temperature Annealing**: Gradual cooling for improved convergence

## Maxwell-Boltzmann Brain

The Maxwell-Boltzmann Brain is based on classical statistical mechanics and the Maxwell-Boltzmann distribution for classical particles.

### Key Features

- **Classical Statistics**: Uses Maxwell-Boltzmann distribution for activations
- **Energy Normalization**: Optional energy conservation layers
- **Thermal Annealing**: Temperature scheduling during training
- **Thermodynamic Properties**: Computes thermal energy and partition function

### Mathematical Foundation

The Maxwell-Boltzmann distribution describes classical particle energy distributions:

```
f(E) = exp(-E/kT)
```

Where:
- `E` is the energy level (network activations)
- `k` is Boltzmann constant
- `T` is temperature

### Usage Example

```python
from maxwell_boltzmann_brain import MaxwellBoltzmannBrain
import torch

# Create model with energy normalization
model = MaxwellBoltzmannBrain(
    input_size=20,
    hidden_sizes=[64, 32, 16],
    output_size=2,
    temperature=2.0,
    use_energy_norm=True
)

# Forward pass
x = torch.randn(32, 20)
output = model(x)

# Compute thermodynamic properties
thermal_energy = model.compute_thermal_energy()
partition_function = model.compute_partition_function()

# Thermal annealing during training
for epoch in range(100):
    current_temp = model.thermal_anneal(epoch, 100, initial_temp=2.0, final_temp=0.1)
    # ... training code ...
```

### Specialized Architectures

#### Classical Gas Network

A specialized Maxwell-Boltzmann network that simulates classical gas particle behavior:

```python
from maxwell_boltzmann_brain import create_classical_gas_network

gas_model = create_classical_gas_network(
    input_size=15,
    hidden_sizes=[128, 64, 32],
    output_size=3,
    temperature=3.0
)
```

## Examples

The `examples/` directory contains comprehensive demonstrations:

### Running Examples

```bash
# Run Fermi-Dirac brain examples
cd examples
python fermi_dirac_example.py

# Run Maxwell-Boltzmann brain examples
python maxwell_boltzmann_example.py
```

### Example Scripts Include

1. **Classification Tasks**: Binary and multi-class classification
2. **Regression Tasks**: Continuous value prediction
3. **Temperature Analysis**: Performance at different temperatures
4. **Energy Conservation**: Demonstration of thermodynamic properties
5. **Annealing Strategies**: Temperature scheduling during training

## Theory

### Fermi-Dirac Statistics vs Maxwell-Boltzmann Statistics

| Property | Fermi-Dirac | Maxwell-Boltzmann |
|----------|-------------|-------------------|
| **Particle Type** | Fermions (half-integer spin) | Classical particles |
| **Exclusion Principle** | Pauli exclusion applies | No exclusion principle |
| **Distribution Shape** | Sigmoid-like, bounded [0,1] | Exponential decay |
| **Temperature Effect** | Sharp transitions at low T | Smooth exponential at all T |
| **Neural Network Analogy** | Quantum computing inspiration | Classical thermodynamics |

### Why These Distributions?

1. **Natural Regularization**: Both distributions provide inherent bounds on activations
2. **Temperature Control**: Temperature acts as a learnable hyperparameter
3. **Physical Intuition**: Leverage well-understood physical principles
4. **Novel Architectures**: Explore new activation function families

### Applications

- **Quantum Machine Learning**: Fermi-Dirac brains for quantum-inspired algorithms
- **Optimization**: Thermal annealing for better convergence
- **Ensemble Methods**: Combining quantum and classical statistical approaches
- **Research**: Novel activation functions and initialization schemes

## API Reference

### Fermi-Dirac Brain API

#### `FermiDiracBrain(input_size, hidden_sizes, output_size, temperature=1.0, dropout_rate=0.1)`

Main neural network class implementing Fermi-Dirac statistics.

**Parameters:**
- `input_size` (int): Number of input features
- `hidden_sizes` (list): List of hidden layer sizes
- `output_size` (int): Number of output units
- `temperature` (float): Initial temperature parameter
- `dropout_rate` (float): Dropout probability

**Methods:**
- `forward(x)`: Forward pass through the network
- `compute_quantum_entropy()`: Calculate quantum entropy
- `get_fermi_energy()`: Get average chemical potential
- `adjust_temperature(new_temp)`: Change temperature of all layers

#### `FermiDiracActivation(temperature=1.0, chemical_potential=0.0)`

Fermi-Dirac activation function.

#### `FermiDiracLayer(in_features, out_features, temperature=1.0, use_bias=True)`

Single layer with Fermi-Dirac activation.

### Maxwell-Boltzmann Brain API

#### `MaxwellBoltzmannBrain(input_size, hidden_sizes, output_size, temperature=1.0, use_energy_norm=True, dropout_rate=0.1)`

Main neural network class implementing Maxwell-Boltzmann statistics.

**Parameters:**
- `input_size` (int): Number of input features
- `hidden_sizes` (list): List of hidden layer sizes
- `output_size` (int): Number of output units
- `temperature` (float): Initial temperature parameter
- `use_energy_norm` (bool): Whether to use energy normalization
- `dropout_rate` (float): Dropout probability

**Methods:**
- `forward(x)`: Forward pass through the network
- `compute_thermal_energy()`: Calculate average thermal energy
- `compute_partition_function()`: Calculate partition function
- `thermal_anneal(current_epoch, total_epochs, initial_temp, final_temp)`: Thermal annealing
- `adjust_temperature(new_temp)`: Change temperature of all layers

#### `MaxwellBoltzmannActivation(temperature=1.0, energy_shift=0.0)`

Maxwell-Boltzmann activation function.

#### `EnergyNormalization(num_features, eps=1e-5)`

Energy conservation normalization layer.

### Utility Functions

```python
# Create pre-configured models
create_fermi_dirac_classifier(input_size, num_classes, hidden_sizes, temperature)
create_fermi_dirac_regressor(input_size, output_size, hidden_sizes, temperature)
create_maxwell_boltzmann_classifier(input_size, num_classes, hidden_sizes, temperature)
create_maxwell_boltzmann_regressor(input_size, output_size, hidden_sizes, temperature)
create_classical_gas_network(input_size, hidden_sizes, output_size, temperature)
```

## Performance Characteristics

### Temperature Effects

- **Low Temperature (T < 0.5)**: Sharp, decisive activations; may lead to vanishing gradients
- **Medium Temperature (T = 1.0)**: Balanced activation dynamics
- **High Temperature (T > 2.0)**: Smooth, diffuse activations; good for exploration

### Computational Complexity

Both networks have similar complexity to standard neural networks with additional overhead for:
- Temperature parameter management
- Statistical property computation
- Energy normalization (Maxwell-Boltzmann only)

## Contributing

We welcome contributions! Please see our contribution guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black src/ examples/

# Lint code
flake8 src/ examples/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use these neural networks in your research, please cite:

```bibtex
@misc{statistical_physics_nn_2024,
  title={Statistical Physics-Inspired Neural Networks: Fermi-Dirac and Maxwell-Boltzmann Brains},
  author={Not-trained Neural Networks Notes},
  year={2024},
  url={https://github.com/ewdlop/Not-trained-Neural-Networks-Notes}
}
```

## Acknowledgments

- Inspired by fundamental principles in quantum mechanics and statistical physics
- Built using PyTorch framework
- Examples use scikit-learn for data generation

## Further Reading

- [Fermi-Dirac Statistics](https://en.wikipedia.org/wiki/Fermi%E2%80%93Dirac_statistics)
- [Maxwell-Boltzmann Statistics](https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_statistics)
- [Statistical Mechanics in Machine Learning](https://arxiv.org/abs/physics/0001057)
- [Quantum Machine Learning](https://arxiv.org/abs/1611.09347)
