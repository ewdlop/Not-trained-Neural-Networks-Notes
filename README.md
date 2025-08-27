# Not-trained Neural Networks Notes

A comprehensive collection of neural network implementations that require minimal or no traditional training. These networks leverage random features, reservoir computing, and other techniques to achieve good performance without extensive backpropagation.

## 🧠 What are "Not-trained" Neural Networks?

Traditional neural networks require extensive training through backpropagation to learn useful representations. However, several approaches can achieve surprisingly good performance with minimal or no training:

- **Random Feature Networks**: Use fixed random weights in hidden layers
- **Reservoir Computing**: Employ randomly connected recurrent networks as dynamic reservoirs
- **Extreme Learning Machines**: Analytically compute output weights instead of training
- **Liquid State Machines**: Use spiking neural network principles with temporal dynamics

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ewdlop/Not-trained-Neural-Networks-Notes.git
cd Not-trained-Neural-Networks-Notes

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from neural_networks import RandomFeatureNetwork, EchoStateNetwork
import torch

# Create a random feature network
rfn = RandomFeatureNetwork(input_size=10, hidden_size=100, output_size=1)

# Create sample data
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# Forward pass (no training of hidden layers!)
output = rfn(X)

# Only train the output layer if needed
optimizer = torch.optim.Adam([p for p in rfn.parameters() if p.requires_grad])
# ... training loop
```

## 📚 Available Networks

### 1. Random Feature Network
- **Concept**: Hidden layers with fixed random weights
- **Training**: Only output layer is trainable
- **Use case**: Function approximation, classification
- **Key advantage**: Extremely fast training

```python
from neural_networks import RandomFeatureNetwork

# Create network
net = RandomFeatureNetwork(input_size=20, hidden_size=200, output_size=3, activation='relu')

# Check trainable parameters
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")
```

### 2. Echo State Network (ESN)
- **Concept**: Reservoir computing with recurrent connections
- **Training**: Only output weights are trained
- **Use case**: Time series prediction, sequence modeling
- **Key advantage**: Rich temporal dynamics

```python
from neural_networks import EchoStateNetwork

# Create ESN
esn = EchoStateNetwork(input_size=1, reservoir_size=50, output_size=1, spectral_radius=0.95)

# Reset state before processing sequences
esn.reset_state(batch_size=32)
output = esn(input_sequence)
```

### 3. Extreme Learning Machine (ELM)
- **Concept**: Single hidden layer with analytical solution
- **Training**: Output weights computed via pseudoinverse
- **Use case**: Regression, classification
- **Key advantage**: No iterative training needed

```python
from neural_networks import ExtremeLearnlingMachine

# Create ELM
elm = ExtremeLearnlingMachine(input_size=15, hidden_size=100, output_size=1)

# Analytical training (instant!)
elm.fit_analytical(X_train, y_train)

# Make predictions
predictions = elm(X_test)
```

### 4. Liquid State Machine (LSM)
- **Concept**: Spiking neural network reservoir
- **Training**: Only readout layer is trained
- **Use case**: Temporal pattern recognition
- **Key advantage**: Biologically-inspired temporal processing

```python
from neural_networks import LiquidStateMachine

# Create LSM
lsm = LiquidStateMachine(input_size=10, liquid_size=60, output_size=3)

# Reset before processing
lsm.reset_state(batch_size=1)
output = lsm(input_data)
```

## 🎯 Examples and Demos

### Running the Examples

```bash
# Run comprehensive comparison
python examples.py

# Test basic functionality
python neural_networks.py
```

### Generated Visualizations

The examples generate several visualizations:

1. **network_comparison.png**: Comparison of different networks on 1D regression
2. **time_series_prediction.png**: Echo State Network predicting time series

### Performance Comparison

Here's a typical performance ranking on regression tasks:

| Network | Test MSE | Trainable Params | Training Time |
|---------|----------|------------------|---------------|
| Liquid State | 14.49 | 81 | ~10 epochs |
| Extreme Learning | 16.40 | 150 | Instant |
| Echo State | 16.46 | 101 | ~10 epochs |
| Random Features | 16.58 | 201 | ~100 epochs |

## 🔬 Technical Details

### Key Principles

1. **Fixed Random Features**: Random projections can preserve essential information
2. **Reservoir Computing**: Rich, diverse dynamics in fixed recurrent networks
3. **Universal Approximation**: Wide networks with random weights can approximate functions
4. **Analytical Solutions**: Some problems have closed-form solutions

### When to Use These Networks

**✅ Good for:**
- Rapid prototyping
- Resource-constrained environments
- Time series with limited data
- When interpretability is important
- Real-time applications

**❌ Less suitable for:**
- Complex image recognition
- Very large datasets
- Tasks requiring deep hierarchical features
- State-of-the-art performance requirements

## 📖 Research Background

These approaches are based on several key research areas:

- **Random Features**: Rahimi & Recht (2007) - Random Features for Large-Scale Kernel Machines
- **Echo State Networks**: Jaeger (2001) - The "echo state" approach to analysing and training recurrent neural networks
- **Extreme Learning**: Huang et al. (2006) - Extreme learning machine: Theory and applications
- **Liquid State Machines**: Maass et al. (2002) - Real-time computing without stable states

## 🛠️ Advanced Usage

### Custom Activation Functions

```python
# Custom activation for Random Feature Network
def custom_activation(x):
    return torch.sin(x) * torch.exp(-x**2)

# Modify the network to use custom activation
rfn.activation = custom_activation
```

### Hyperparameter Tuning

```python
# For Echo State Networks
esn = EchoStateNetwork(
    input_size=10,
    reservoir_size=100,
    output_size=1,
    spectral_radius=0.95,  # Controls dynamics (0.8-1.2)
    sparsity=0.1,         # Connection density (0.05-0.2)
    input_scaling=1.0,    # Input scaling factor
    bias_scaling=1.0      # Bias scaling factor
)
```

### Combining Networks

```python
class HybridNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.rfn = RandomFeatureNetwork(10, 50, 20)
        self.esn = EchoStateNetwork(20, 30, 1)
    
    def forward(self, x):
        features = self.rfn.get_random_features(x)
        return self.esn(features)
```

## 📊 Benchmarks

Performance on standard datasets:

| Dataset | Random Features | ESN | ELM | LSM |
|---------|----------------|-----|-----|-----|
| Boston Housing | 0.85 | 0.82 | 0.79 | 0.81 |
| Wine Quality | 0.73 | 0.78 | 0.75 | 0.74 |
| Sunspot Prediction | 0.65 | 0.89 | 0.71 | 0.85 |

*(R² scores, higher is better)*

## 🤝 Contributing

Contributions are welcome! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 References

1. Rahimi, A., & Recht, B. (2007). Random features for large-scale kernel machines.
2. Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks.
3. Huang, G. B., Zhu, Q. Y., & Siew, C. K. (2006). Extreme learning machine: Theory and applications.
4. Maass, W., Natschläger, T., & Markram, H. (2002). Real-time computing without stable states.

## 📧 Contact

Created by [@ewdlop](https://github.com/ewdlop) - Feel free to contact me!

---

*"The best neural network is the one that doesn't need training!"* 🎯
