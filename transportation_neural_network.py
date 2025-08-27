"""
Transportation Neural Network in PyTorch
========================================

This implementation models the biological neural activation process described in
the 7-stage neural activation cycle:

1. Resting State (静息状态) - -70mV
2. Stimulus Input (刺激输入) - synaptic transmission  
3. Threshold & Depolarization (阈值与去极化) - -55mV threshold
4. Repolarization & Hyperpolarization (再极化与超极化) - K+ channels
5. Refractory Period (不应期) - absolute and relative
6. Signal Propagation (讯号传导) - axon transmission
7. Synaptic Release (突觸釋放) - Ca2+ channels

The network simulates the transportation of neural signals through these stages
using PyTorch tensors and neural network layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List, Optional


class NeuralActivationStage(nn.Module):
    """Base class for neural activation stages"""
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning output and internal state"""
        raise NotImplementedError


class RestingState(NeuralActivationStage):
    """Stage 1: Resting State (-70mV)
    
    Models the neuron's resting potential maintained by Na+/K+ pump
    """
    
    def __init__(self, size: int):
        super().__init__(size, size)
        self.resting_potential = -70.0  # mV
        self.register_buffer('resting_bias', torch.full((size,), self.resting_potential))
        
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initialize to resting potential if no input
        if state is None:
            state = self.resting_bias.clone()
        
        # Decay towards resting potential
        decay_rate = 0.1
        output = state + decay_rate * (self.resting_bias - state)
        return output, output


class SynapticInput(NeuralActivationStage):
    """Stage 2: Stimulus Input - Synaptic transmission
    
    Models excitatory (glutamate) and inhibitory (GABA) inputs
    """
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        self.excitatory = nn.Linear(input_size, output_size)
        self.inhibitory = nn.Linear(input_size, output_size)
        
        # Initialize weights for biological realism
        nn.init.normal_(self.excitatory.weight, 0, 0.1)
        nn.init.normal_(self.inhibitory.weight, 0, 0.1)
        
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            state = torch.zeros(self.output_size)
            
        # Separate excitatory and inhibitory components
        excitatory_input = torch.relu(x)  # Positive inputs are excitatory
        inhibitory_input = torch.relu(-x)  # Negative inputs are inhibitory
        
        excitation = self.excitatory(excitatory_input)
        inhibition = -self.inhibitory(inhibitory_input)  # Negative contribution
        
        output = state + excitation + inhibition
        return output, output


class ThresholdDepolarization(NeuralActivationStage):
    """Stage 3: Threshold & Depolarization (-55mV threshold)
    
    Models voltage-gated Na+ channels and action potential initiation
    """
    
    def __init__(self, size: int):
        super().__init__(size, size)
        self.threshold = -55.0  # mV
        self.action_potential_peak = 30.0  # mV
        
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            state = torch.zeros(self.input_size)
            
        # Check if threshold is reached
        above_threshold = x > self.threshold
        
        # Rapid depolarization for neurons above threshold
        output = torch.where(
            above_threshold,
            torch.full_like(x, self.action_potential_peak),
            x
        )
        
        return output, output


class RepolarizationHyperpolarization(NeuralActivationStage):
    """Stage 4: Repolarization & Hyperpolarization
    
    Models K+ channel opening and membrane potential recovery
    """
    
    def __init__(self, size: int):
        super().__init__(size, size)
        self.k_channel = nn.Parameter(torch.ones(size) * 0.3)  # K+ channel conductance
        self.hyperpolarization_level = -80.0  # mV
        
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            state = torch.zeros(self.input_size)
            
        # K+ channel activation causes repolarization
        k_current = self.k_channel * x
        
        # Rapid repolarization, potential hyperpolarization
        output = x - k_current
        
        # Some neurons may hyperpolarize below resting potential
        hyperpolarized = output < self.hyperpolarization_level
        output = torch.where(hyperpolarized, 
                           torch.full_like(output, self.hyperpolarization_level),
                           output)
        
        return output, output


class RefractoryPeriod(NeuralActivationStage):
    """Stage 5: Refractory Period
    
    Models absolute and relative refractory periods
    """
    
    def __init__(self, size: int):
        super().__init__(size, size)
        self.register_buffer('refractory_counter', torch.zeros(size))
        self.absolute_period = 3  # time steps
        self.relative_period = 7  # time steps
        
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            self.refractory_counter.zero_()
            
        # Detect action potentials (high positive values)
        action_potential = x > 20.0
        
        # Update refractory counter
        self.refractory_counter = torch.where(
            action_potential,
            torch.full_like(self.refractory_counter, float(self.relative_period)),
            torch.clamp(self.refractory_counter - 1, min=0)
        )
        
        # Apply refractory effects
        in_absolute = self.refractory_counter > (self.relative_period - self.absolute_period)
        in_relative = (self.refractory_counter > 0) & (~in_absolute)
        
        output = torch.where(
            in_absolute,
            torch.zeros_like(x),  # Complete block
            torch.where(
                in_relative,
                x * 0.3,  # Reduced response
                x
            )
        )
        
        return output, self.refractory_counter


class SignalPropagation(NeuralActivationStage):
    """Stage 6: Signal Propagation along axon
    
    Models saltatory conduction and axonal transmission
    """
    
    def __init__(self, size: int, propagation_length: int = 10):
        super().__init__(size, size)
        self.propagation_length = propagation_length
        
        # Myelinated vs unmyelinated conduction
        self.myelin_mask = nn.Parameter(torch.bernoulli(torch.full((size,), 0.7)))
        self.conduction_velocity = nn.Parameter(torch.ones(size))
        
        # Buffer for signal propagation delay
        self.register_buffer('signal_buffer', 
                           torch.zeros(propagation_length, size))
        
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shift signal buffer (propagation delay)
        self.signal_buffer = torch.roll(self.signal_buffer, shifts=1, dims=0)
        self.signal_buffer[0] = x
        
        # Myelinated axons conduct faster (saltatory conduction)
        myelinated_delay = 2
        unmyelinated_delay = 5
        
        myelinated_output = self.signal_buffer[myelinated_delay]
        unmyelinated_output = self.signal_buffer[unmyelinated_delay]
        
        output = torch.where(
            self.myelin_mask.bool(),
            myelinated_output,
            unmyelinated_output
        )
        
        # Apply conduction velocity scaling
        output = output * self.conduction_velocity
        
        return output, self.signal_buffer[-1]


class SynapticRelease(NeuralActivationStage):
    """Stage 7: Synaptic Release (Ca2+ channels)
    
    Models neurotransmitter release at synaptic terminals
    """
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        self.ca_channel = nn.Linear(input_size, input_size)
        self.vesicle_release = nn.Linear(input_size, output_size)
        
        # Ca2+ channel has sigmoidal activation
        self.ca_threshold = 10.0
        
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ca2+ channel activation by action potential
        ca_activation = torch.sigmoid((x - self.ca_threshold) / 5.0)
        ca_influx = self.ca_channel(x) * ca_activation
        
        # Vesicle release probability depends on Ca2+ concentration
        release_probability = torch.sigmoid(ca_influx)
        neurotransmitter = self.vesicle_release(release_probability)
        
        return neurotransmitter, ca_influx


class TransportationNeuralNetwork(nn.Module):
    """Complete Transportation Neural Network
    
    Chains all 7 stages of neural activation in sequence to simulate
    the complete neural signal transportation process.
    """
    
    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 5):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize all 7 stages
        self.stage1_resting = RestingState(hidden_size)
        self.stage2_synaptic = SynapticInput(input_size, hidden_size)
        self.stage3_threshold = ThresholdDepolarization(hidden_size)
        self.stage4_repolarization = RepolarizationHyperpolarization(hidden_size)
        self.stage5_refractory = RefractoryPeriod(hidden_size)
        self.stage6_propagation = SignalPropagation(hidden_size)
        self.stage7_release = SynapticRelease(hidden_size, output_size)
        
        # Store intermediate states for visualization
        self.stage_outputs = {}
        self.stage_states = {}
        
    def forward(self, x: torch.Tensor, reset_state: bool = False) -> torch.Tensor:
        """Forward pass through all neural activation stages"""
        
        if reset_state:
            self.reset_states()
        
        # Stage 1: Resting State
        resting_output, resting_state = self.stage1_resting(x)
        self.stage_outputs['resting'] = resting_output
        self.stage_states['resting'] = resting_state
        
        # Stage 2: Synaptic Input
        synaptic_output, synaptic_state = self.stage2_synaptic(x, resting_output)
        self.stage_outputs['synaptic'] = synaptic_output
        self.stage_states['synaptic'] = synaptic_state
        
        # Stage 3: Threshold & Depolarization
        threshold_output, threshold_state = self.stage3_threshold(synaptic_output)
        self.stage_outputs['threshold'] = threshold_output
        self.stage_states['threshold'] = threshold_state
        
        # Stage 4: Repolarization
        repolar_output, repolar_state = self.stage4_repolarization(threshold_output)
        self.stage_outputs['repolarization'] = repolar_output
        self.stage_states['repolarization'] = repolar_state
        
        # Stage 5: Refractory Period
        refractory_output, refractory_state = self.stage5_refractory(repolar_output)
        self.stage_outputs['refractory'] = refractory_output
        self.stage_states['refractory'] = refractory_state
        
        # Stage 6: Signal Propagation
        propagation_output, propagation_state = self.stage6_propagation(refractory_output)
        self.stage_outputs['propagation'] = propagation_output
        self.stage_states['propagation'] = propagation_state
        
        # Stage 7: Synaptic Release
        release_output, release_state = self.stage7_release(propagation_output)
        self.stage_outputs['release'] = release_output
        self.stage_states['release'] = release_state
        
        return release_output
    
    def reset_states(self):
        """Reset all internal states"""
        self.stage_outputs.clear()
        self.stage_states.clear()
        
        # Reset refractory counters
        if hasattr(self.stage5_refractory, 'refractory_counter'):
            self.stage5_refractory.refractory_counter.zero_()
            
        # Reset propagation buffer
        if hasattr(self.stage6_propagation, 'signal_buffer'):
            self.stage6_propagation.signal_buffer.zero_()
    
    def get_stage_outputs(self) -> dict:
        """Get outputs from all stages for visualization"""
        return self.stage_outputs.copy()
    
    def visualize_activation(self, input_sequence: torch.Tensor, title: str = "Neural Activation Process"):
        """Visualize the neural activation process over time"""
        
        plt.figure(figsize=(15, 10))
        
        # Process sequence
        outputs_over_time = {stage: [] for stage in ['resting', 'synaptic', 'threshold', 
                                                   'repolarization', 'refractory', 
                                                   'propagation', 'release']}
        
        self.reset_states()
        
        for t, input_t in enumerate(input_sequence):
            self.forward(input_t.unsqueeze(0))
            stage_outputs = self.get_stage_outputs()
            
            for stage_name, output in stage_outputs.items():
                outputs_over_time[stage_name].append(output.mean().item())
        
        # Plot each stage
        stage_names = list(outputs_over_time.keys())
        for i, stage_name in enumerate(stage_names):
            plt.subplot(3, 3, i+1)
            plt.plot(outputs_over_time[stage_name], linewidth=2)
            plt.title(f"Stage {i+1}: {stage_name.title()}")
            plt.xlabel("Time Steps")
            plt.ylabel("Activation (mV)")
            plt.grid(True, alpha=0.3)
        
        # Overall summary plot
        plt.subplot(3, 3, 8)
        for stage_name in stage_names:
            plt.plot(outputs_over_time[stage_name], label=stage_name, alpha=0.7)
        plt.title("All Stages Combined")
        plt.xlabel("Time Steps")
        plt.ylabel("Activation (mV)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16, y=1.02)
        return plt.gcf()


def create_sample_stimuli(num_steps: int = 50, input_size: int = 10) -> torch.Tensor:
    """Create sample stimuli patterns for testing"""
    
    # Pattern 1: Gradual increase (depolarizing stimulus)
    gradual = torch.linspace(-80, -50, num_steps).unsqueeze(1).repeat(1, input_size)
    
    # Pattern 2: Sharp spike (action potential trigger)
    spike = torch.full((num_steps, input_size), -70.0)
    spike[15:20] = -40.0  # Strong depolarizing pulse
    
    # Pattern 3: Oscillatory pattern (rhythmic input)
    t = torch.linspace(0, 4*np.pi, num_steps)
    oscillatory = -70 + 15 * torch.sin(t).unsqueeze(1).repeat(1, input_size)
    
    # Pattern 4: Random noise (background activity)
    random_noise = -70 + torch.randn(num_steps, input_size) * 5
    
    return {
        'gradual': gradual,
        'spike': spike, 
        'oscillatory': oscillatory,
        'random': random_noise
    }


if __name__ == "__main__":
    # Demonstration of the Transportation Neural Network
    print("Transportation Neural Network Demo")
    print("=" * 50)
    
    # Create network
    network = TransportationNeuralNetwork(input_size=10, hidden_size=20, output_size=5)
    
    # Generate test stimuli
    stimuli = create_sample_stimuli(num_steps=50, input_size=10)
    
    print(f"Network architecture:")
    print(f"- Input size: {network.input_size}")
    print(f"- Hidden size: {network.hidden_size}")
    print(f"- Output size: {network.output_size}")
    print(f"- Number of parameters: {sum(p.numel() for p in network.parameters())}")
    
    # Test with different stimuli patterns
    for pattern_name, stimulus in stimuli.items():
        print(f"\nTesting with {pattern_name} stimulus...")
        
        # Run forward pass
        outputs = []
        network.reset_states()
        
        for t in range(stimulus.shape[0]):
            output = network(stimulus[t])
            outputs.append(output.detach())
        
        final_output = torch.stack(outputs)
        print(f"Output shape: {final_output.shape}")
        print(f"Output range: [{final_output.min().item():.2f}, {final_output.max().item():.2f}]")
        
        # Create visualization
        fig = network.visualize_activation(stimulus, f"Neural Activation - {pattern_name.title()} Stimulus")
        plt.savefig(f'/tmp/neural_activation_{pattern_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved: /tmp/neural_activation_{pattern_name}.png")
    
    print("\nTransportation Neural Network demonstration completed!")
    print("The network successfully models the 7-stage neural activation process:")
    print("1. Resting State → 2. Synaptic Input → 3. Threshold & Depolarization")
    print("4. Repolarization → 5. Refractory Period → 6. Signal Propagation → 7. Synaptic Release")