"""
Simple example demonstrating the Transportation Neural Network

This script shows basic usage of the neural activation model.
"""

import torch
import matplotlib.pyplot as plt
from transportation_neural_network import TransportationNeuralNetwork, create_sample_stimuli

def basic_example():
    """Basic usage example"""
    print("Basic Transportation Neural Network Example")
    print("=" * 50)
    
    # Create a simple network
    network = TransportationNeuralNetwork(input_size=5, hidden_size=10, output_size=3)
    
    # Create a simple stimulus (action potential-like input)
    stimulus = torch.full((30, 5), -70.0)  # Start at resting potential
    stimulus[10:15] = -40.0  # Strong depolarizing pulse
    
    print(f"Input stimulus shape: {stimulus.shape}")
    print(f"Network parameters: {sum(p.numel() for p in network.parameters())}")
    
    # Process the stimulus
    network.reset_states()
    outputs = []
    
    for t in range(stimulus.shape[0]):
        output = network(stimulus[t])
        outputs.append(output.detach())
    
    final_outputs = torch.stack(outputs)
    print(f"Output shape: {final_outputs.shape}")
    
    # Show the neural activation stages
    print("\nStage-by-stage activation for last timestep:")
    stage_outputs = network.get_stage_outputs()
    for stage_name, output in stage_outputs.items():
        print(f"  {stage_name:15}: mean={output.mean().item():6.2f}, "
              f"std={output.std().item():6.2f}")
    
    return network, stimulus, final_outputs

def demonstrate_refractory_period():
    """Demonstrate the refractory period effect"""
    print("\nRefractory Period Demonstration")
    print("=" * 40)
    
    network = TransportationNeuralNetwork(input_size=3, hidden_size=5, output_size=2)
    
    # Create two action potentials close together
    stimulus = torch.full((20, 3), -70.0)
    stimulus[5:7] = -30.0   # First action potential
    stimulus[8:10] = -30.0  # Second action potential (should be reduced)
    
    network.reset_states()
    outputs = []
    refractory_states = []
    
    for t in range(stimulus.shape[0]):
        output = network(stimulus[t])
        outputs.append(output.mean().item())
        
        # Get refractory state
        if 'refractory' in network.stage_states:
            refractory_states.append(network.stage_states['refractory'].mean().item())
        else:
            refractory_states.append(0)
    
    print("Time step | Stimulus | Output | Refractory")
    print("-" * 45)
    for t in range(len(outputs)):
        stim_val = stimulus[t].mean().item()
        print(f"{t:8d} | {stim_val:7.1f} | {outputs[t]:6.3f} | {refractory_states[t]:10.1f}")
    
    return outputs, refractory_states

if __name__ == "__main__":
    # Run basic example
    network, stimulus, outputs = basic_example()
    
    # Run refractory period demonstration
    demo_outputs, demo_refractory = demonstrate_refractory_period()
    
    print("\nExample completed successfully!")
    print("The Transportation Neural Network models the complete")
    print("biological neural activation process in PyTorch.")