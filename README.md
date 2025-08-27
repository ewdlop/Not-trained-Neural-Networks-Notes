# Not-trained-Neural-Networks-Notes

This repository contains educational implementations of neural network concepts, focusing on biological neural processes and their computational models.

## Transportation Neural Network in PyTorch

A comprehensive implementation of the biological neural activation process in PyTorch, modeling the complete 7-stage neural signal transportation cycle.

### Neural Activation Process (神經活化過程)

The Transportation Neural Network models the biological neural activation process through seven distinct stages:

1. **Resting State (靜息狀態)** - Maintains resting potential at -70mV
2. **Stimulus Input (刺激輸入)** - Synaptic transmission with excitatory/inhibitory inputs
3. **Threshold & Depolarization (閾值與去極化)** - Action potential initiation at -55mV threshold
4. **Repolarization & Hyperpolarization (再極化與超極化)** - K+ channel activation
5. **Refractory Period (不應期)** - Absolute and relative refractory periods
6. **Signal Propagation (訊號傳導)** - Axonal transmission with myelinated/unmyelinated conduction
7. **Synaptic Release (突觸釋放)** - Ca2+ channel-mediated neurotransmitter release

### Features

- **Biologically Accurate**: Models actual neural membrane dynamics and ion channel behavior
- **Modular Design**: Each activation stage is implemented as a separate PyTorch module
- **Visualization**: Built-in plotting capabilities for observing neural activation patterns
- **Educational**: Comprehensive documentation linking biological processes to computational implementation

### Quick Start

```python
from transportation_neural_network import TransportationNeuralNetwork, create_sample_stimuli

# Create network
network = TransportationNeuralNetwork(input_size=10, hidden_size=20, output_size=5)

# Generate test stimulus
stimuli = create_sample_stimuli(num_steps=50, input_size=10)

# Process neural signal
network.reset_states()
for t in range(stimuli['spike'].shape[0]):
    output = network(stimuli['spike'][t])

# Visualize activation process
fig = network.visualize_activation(stimuli['spike'], "Neural Activation Process")
```

### Files

- `transportation_neural_network.py` - Main implementation with all 7 stages
- `example_usage.py` - Simple usage examples and demonstrations
- `neural_activation_spike.png` - Example visualization output

### Requirements

- PyTorch >= 2.0
- matplotlib
- numpy

### Installation

```bash
pip install torch matplotlib numpy
```

### Usage Examples

See `example_usage.py` for basic examples, or run the main script:

```bash
python transportation_neural_network.py
```

This will generate visualizations showing how different stimulus patterns (gradual, spike, oscillatory, random) propagate through the neural activation stages.

### Educational Value

This implementation serves as an educational tool for understanding:
- How biological neurons process electrical signals
- The relationship between neural membrane dynamics and computation
- Signal propagation and synaptic transmission mechanisms
- Refractory periods and their computational implications
- The connection between neuroscience and artificial neural networks

### Technical Details

The network uses PyTorch tensors to represent membrane potentials and implements realistic neural dynamics including:
- Voltage-gated ion channels (Na+, K+, Ca2+)
- Synaptic transmission mechanisms
- Action potential propagation delays
- Refractory period effects
- Myelinated vs unmyelinated conduction

Each stage maintains internal state and can be used independently or as part of the complete neural activation pipeline.
