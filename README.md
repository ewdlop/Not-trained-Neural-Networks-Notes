# Not-trained-Neural-Networks-Notes

## Super-Ego-Id Neural Network in PyTorch

This repository contains an implementation of a neural network architecture inspired by Freudian psychoanalytic theory. The network consists of three distinct components representing the psychological concepts of Id, Ego, and Super-ego.

### Architecture Overview

The **Super-Ego-Id Neural Network** demonstrates how psychological concepts can be translated into neural network design:

#### Components

1. **Id Network**: Represents primitive, instinctual responses
   - Operates on immediate gratification principles
   - Uses higher activation functions to represent impulsive behavior
   - Minimal constraints on outputs

2. **Ego Network**: Represents realistic, mediating responses  
   - Balances between id and super-ego demands
   - Uses batch normalization for stability
   - Moderate activation constraints

3. **Super-ego Network**: Represents moral and ethical constraints
   - Applies learned social rules and moral guidelines
   - Uses constrained activation functions
   - Additional constraint layers to limit outputs

#### Integration Mechanism

- **Attention Mechanism**: Multi-head attention to dynamically weight component contributions
- **Integration Layer**: Combines outputs from all three components
- **Component Analysis**: Tools to analyze the dominance of each psychological component

### Files

- `super_ego_id_network.py`: Main implementation of the neural network architecture
- `example_usage.py`: Comprehensive example demonstrating training and analysis
- `requirements.txt`: Python dependencies

### Usage

#### Basic Usage

```python
from super_ego_id_network import SuperEgoIdNetwork

# Create network
network = SuperEgoIdNetwork(input_dim=20, hidden_dim=64, output_dim=10)

# Forward pass
input_data = torch.randn(32, 20)
output, components = network(input_data)

# Analyze psychological components
analysis = network.get_component_analysis(input_data)
print(f"Id dominance: {analysis['id_dominance']:.3f}")
print(f"Ego dominance: {analysis['ego_dominance']:.3f}")  
print(f"Super-ego dominance: {analysis['superego_dominance']:.3f}")
```

#### Running the Example

```bash
# Install dependencies
pip install -r requirements.txt

# Run the basic demonstration
python super_ego_id_network.py

# Run the comprehensive example with training
python example_usage.py
```

### Key Features

- **Psychological Interpretation**: Each component has distinct characteristics reflecting psychological theory
- **Attention Mechanism**: Dynamic weighting of psychological components
- **Component Analysis**: Tools to understand which psychological aspect dominates decisions
- **Educational Design**: Clear documentation and examples for learning purposes
- **Extensible Architecture**: Easy to modify for different applications

### Theoretical Background

The implementation is based on Freudian structural model of the psyche:

- **Id**: The pleasure principle - seeks immediate gratification
- **Ego**: The reality principle - mediates between id and super-ego  
- **Super-ego**: The moral principle - represents internalized social rules

### Applications

This architecture can be useful for:

- **Behavioral Modeling**: Understanding decision-making processes
- **AI Ethics**: Incorporating moral constraints into AI systems
- **Educational Purposes**: Demonstrating psychology-inspired neural architectures
- **Research**: Exploring multi-component neural network designs

### Installation

```bash
git clone https://github.com/ewdlop/Not-trained-Neural-Networks-Notes.git
cd Not-trained-Neural-Networks-Notes
pip install -r requirements.txt
```

### Contributing

This is an educational repository demonstrating psychological concepts in neural network design. Contributions that enhance the educational value or extend the psychological modeling aspects are welcome.
