# Not-trained-Neural-Networks-Notes

This repository contains a Type Theory Neural Network implementation in PyTorch that demonstrates the application of type theory concepts to deep learning.

## Type Theory Neural Network

The implementation includes a neural network that:

1. **Uses strong typing throughout the computation graph** - Every tensor carries type information as it flows through the network
2. **Performs runtime type checking** - Layers validate that they receive the expected input types
3. **Tracks type evolution** - The network maintains a trace of how types transform through each layer
4. **Demonstrates dependent types** - Types that depend on the network architecture and data flow

### Key Features

- **TypedTensor**: A wrapper around PyTorch tensors that includes type information
- **TypedLayer**: Abstract base class for layers that preserve and transform type information
- **TypedLinear**: Linear layers with explicit input/output type declarations
- **TypedActivation**: Activation functions that transform types according to their mathematical properties
- **TypeTheoryNeuralNetwork**: A complete neural network that tracks type evolution

### Example Usage

```python
from type_theory_nn import create_classification_network

# Create a classification network with proper type annotations
net = create_classification_network(
    input_size=10,
    hidden_sizes=[20, 15],
    num_classes=5
)

# Get the type signature
print(f"Network type signature: {net.get_type_signature()}")

# Forward pass with type tracking
import torch
sample_input = torch.randn(8, 10)
output, type_trace = net(sample_input, "FeatureVector")

print(f"Type evolution: {' -> '.join(type_trace)}")
```

### Type System

The type system tracks several categories of types:

- **Input Types**: `FeatureVector`, `InputFeatures`, etc.
- **Intermediate Types**: `Hidden_0`, `Hidden_1`, etc.
- **Activation Types**: `Relu_Hidden_0`, `Tanh_Hidden_1`, etc.
- **Output Types**: `ClassProbabilities`, `RegressionTarget`, `Probability`

### Files

- `type_theory_nn.py` - Main implementation of the type theory neural network
- `demo.py` - Comprehensive demonstration script showing all features
- `requirements.txt` - Dependencies (PyTorch, NumPy, typing_extensions)

### Running the Demo

```bash
pip install -r requirements.txt
python demo.py
```

This will run five different demonstrations:
1. Basic Type Theory Neural Network Usage
2. Classification Network Example
3. Regression Network Example  
4. Individual Typed Components
5. Type Compatibility Checking

### Educational Value

This implementation serves as an educational tool for understanding:
- How type theory can be applied to neural networks
- Runtime type checking in deep learning
- Type-safe neural network architectures
- The relationship between mathematical operations and type transformations

The code heavily uses Python's type system with generics, protocols, and type variables to demonstrate modern type-safe programming practices in the context of machine learning.
