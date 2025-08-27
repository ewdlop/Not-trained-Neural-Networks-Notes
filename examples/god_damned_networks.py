#!/usr/bin/env python3
"""
God-damned Neural Networks: A Collection of Frustrating, Pathological, and Satirical Neural Network Examples

This module demonstrates the most annoying, counterintuitive, and downright infuriating
aspects of neural networks through algebraic implementations that don't require training
but still manage to embody all the worst characteristics of modern ML.

Author: Frustrated ML Researchers Everywhere
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algebraic_neural_network import AlgebraicLayer, AlgebraicNeuralNetwork


class VanishingGradientLayer(AlgebraicLayer):
    """
    A layer that simulates the vanishing gradient problem by making outputs
    exponentially smaller with each application. Because who doesn't love
    gradients that disappear faster than your motivation on Monday morning?
    """
    
    def __init__(self, input_size: int, output_size: int, vanishing_factor: float = 0.1):
        super().__init__(input_size, output_size, "vanishing_gradient")
        self.vanishing_factor = vanishing_factor
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply vanishing gradient simulation - watch your signal disappear!"""
        batch_size = x.shape[0]
        result = np.zeros((batch_size, self.output_size))
        
        for i in range(batch_size):
            for j in range(self.output_size):
                # Simulate deep network signal decay
                signal = np.sum(x[i] * np.arange(1, self.input_size + 1))
                # Apply exponential decay like a deep network's gradient
                result[i, j] = signal * (self.vanishing_factor ** (j + 1))
                
        return result


class ExplodingGradientLayer(AlgebraicLayer):
    """
    The opposite of vanishing gradients - outputs that grow exponentially!
    Perfect for when you want your loss to reach infinity in record time.
    """
    
    def __init__(self, input_size: int, output_size: int, explosion_factor: float = 2.0):
        super().__init__(input_size, output_size, "exploding_gradient")
        self.explosion_factor = explosion_factor
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply exploding gradient simulation - hold onto your computational hat!"""
        batch_size = x.shape[0]
        result = np.zeros((batch_size, self.output_size))
        
        for i in range(batch_size):
            for j in range(self.output_size):
                # Start with input signal
                signal = np.sum(x[i] * np.random.RandomState(42 + j).randn(self.input_size))
                # Apply exponential growth - because bigger is always better, right?
                result[i, j] = signal * (self.explosion_factor ** (j + 1))
                
        return np.tanh(result)  # Clip to prevent actual explosion


class OverfittingLayer(AlgebraicLayer):
    """
    A layer that memorizes every input it's ever seen. Because generalization
    is overrated and we all love models that work perfectly on training data
    and terribly on everything else.
    """
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size, "overfitting")
        self.memory = {}  # The layer's "training data"
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Overfit like it's 1999!"""
        batch_size = x.shape[0]
        result = np.zeros((batch_size, self.output_size))
        
        for i in range(batch_size):
            # Create a "hash" of the input
            input_hash = hash(tuple(x[i].round(3)))  # Round for fuzzy matching
            
            if input_hash in self.memory:
                # We've "seen" this before - perfect memorization!
                result[i] = self.memory[input_hash]
            else:
                # New input - generate random but deterministic output
                np.random.seed(abs(input_hash) % 2**31)
                output = np.random.randn(self.output_size)
                self.memory[input_hash] = output
                result[i] = output
                
        return result


class CatastrophicForgettingLayer(AlgebraicLayer):
    """
    A layer that forgets everything it learned previously when it sees new data.
    Perfect for simulating the frustration of continual learning!
    """
    
    def __init__(self, input_size: int, output_size: int, memory_size: int = 3):
        super().__init__(input_size, output_size, "catastrophic_forgetting")
        self.memory_size = memory_size
        self.recent_inputs = []
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forget everything you knew and learn anew!"""
        batch_size = x.shape[0]
        result = np.zeros((batch_size, self.output_size))
        
        for i in range(batch_size):
            # Add current input to recent memory
            self.recent_inputs.append(x[i].copy())
            
            # Catastrophic forgetting - only remember last few inputs
            if len(self.recent_inputs) > self.memory_size:
                self.recent_inputs.pop(0)
            
            # Output based only on recent memory
            if len(self.recent_inputs) > 0:
                avg_recent = np.mean(self.recent_inputs, axis=0)
                # Transform based on recent memory only
                for j in range(self.output_size):
                    result[i, j] = np.sin(avg_recent[j % self.input_size] * (j + 1))
            
        return result


class ModeCollapseLayer(AlgebraicLayer):
    """
    A layer that always produces the same output regardless of input.
    Like a GAN generator that gave up on diversity and decided one
    output is good enough for everything.
    """
    
    def __init__(self, input_size: int, output_size: int, collapsed_value: float = 0.5):
        super().__init__(input_size, output_size, "mode_collapse")
        self.collapsed_value = collapsed_value
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Why generate variety when you can generate one thing really well?"""
        batch_size = x.shape[0]
        # Everyone gets the same output! Diversity is overrated.
        return np.full((batch_size, self.output_size), self.collapsed_value)


class NonConvergentLayer(AlgebraicLayer):
    """
    A layer whose output oscillates wildly and never settles down.
    Perfect for simulating those training runs that just won't converge.
    """
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size, "non_convergent")
        self.iteration = 0
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Oscillate forever because convergence is for quitters!"""
        self.iteration += 1
        batch_size = x.shape[0]
        result = np.zeros((batch_size, self.output_size))
        
        for i in range(batch_size):
            for j in range(self.output_size):
                # Oscillate based on iteration count and input
                signal = np.sum(x[i]) + self.iteration
                # Create chaotic oscillation
                result[i, j] = np.sin(signal * (j + 1)) * np.cos(self.iteration * 0.1)
                
        return result


def create_god_damned_network() -> AlgebraicNeuralNetwork:
    """
    Create the most frustrating neural network known to humankind.
    This network embodies every pathological behavior that makes
    ML practitioners question their life choices.
    """
    network = AlgebraicNeuralNetwork()
    
    # Start with vanishing gradients
    network.add_layer(VanishingGradientLayer(4, 6, vanishing_factor=0.01))
    
    # Then explode them
    network.add_layer(ExplodingGradientLayer(6, 8, explosion_factor=1.5))
    
    # Overfit to everything
    network.add_layer(OverfittingLayer(8, 6))
    
    # Then forget it all
    network.add_layer(CatastrophicForgettingLayer(6, 4, memory_size=2))
    
    # Collapse to a single mode
    network.add_layer(ModeCollapseLayer(4, 3, collapsed_value=0.42))
    
    # And finally, never converge
    network.add_layer(NonConvergentLayer(3, 2))
    
    return network


def create_mildly_annoying_network() -> AlgebraicNeuralNetwork:
    """
    A network that's just annoying enough to be frustrating but not
    completely broken. Like that bug that only happens in production.
    """
    network = AlgebraicNeuralNetwork()
    
    network.add_layer(VanishingGradientLayer(4, 5, vanishing_factor=0.3))
    network.add_layer(OverfittingLayer(5, 4))
    network.add_layer(NonConvergentLayer(4, 2))
    
    return network


def demonstrate_god_damned_networks():
    """Demonstrate the most frustrating aspects of neural networks."""
    print("😈 God-damned Neural Networks: A Tour of ML Frustrations")
    print("=" * 70)
    
    print("\n🔥 Individual Layer Demonstrations:")
    print("-" * 40)
    
    # Test data
    np.random.seed(42)
    test_input = np.random.randn(3, 4)
    
    # Vanishing gradients
    print("\n1. Vanishing Gradient Layer:")
    print("   Watch your signal disappear into the void...")
    vanishing = VanishingGradientLayer(4, 3, vanishing_factor=0.1)
    output = vanishing.forward(test_input)
    print(f"   Input magnitude: {np.linalg.norm(test_input):.6f}")
    print(f"   Output magnitude: {np.linalg.norm(output):.6f}")
    print(f"   Signal loss: {((np.linalg.norm(test_input) - np.linalg.norm(output)) / np.linalg.norm(test_input) * 100):.2f}%")
    
    # Exploding gradients
    print("\n2. Exploding Gradient Layer:")
    print("   Your gradients are going to the moon! 🚀")
    exploding = ExplodingGradientLayer(4, 3, explosion_factor=2.0)
    output = exploding.forward(test_input * 0.1)  # Small input to prevent actual explosion
    print(f"   Input magnitude: {np.linalg.norm(test_input * 0.1):.6f}")
    print(f"   Output magnitude: {np.linalg.norm(output):.6f}")
    print(f"   Growth factor: {(np.linalg.norm(output) / np.linalg.norm(test_input * 0.1)):.2f}x")
    
    # Overfitting
    print("\n3. Overfitting Layer:")
    print("   Perfect memory, terrible generalization...")
    overfitting = OverfittingLayer(4, 3)
    output1 = overfitting.forward(test_input)
    output2 = overfitting.forward(test_input)  # Same input
    output3 = overfitting.forward(test_input + 0.001)  # Slightly different input
    print(f"   Same input consistency: {np.allclose(output1, output2)}")
    print(f"   Different input similarity: {np.linalg.norm(output1 - output3):.6f}")
    print(f"   Memorized patterns: {len(overfitting.memory)}")
    
    # Mode collapse
    print("\n4. Mode Collapse Layer:")
    print("   Why be diverse when you can be... consistent?")
    mode_collapse = ModeCollapseLayer(4, 3, collapsed_value=0.5)
    diverse_inputs = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [-1, -1, -1, -1],
        [100, -100, 50, -50]
    ])
    outputs = mode_collapse.forward(diverse_inputs)
    print(f"   Input diversity: {np.std(diverse_inputs):.6f}")
    print(f"   Output diversity: {np.std(outputs):.6f}")
    print(f"   All outputs identical: {np.allclose(outputs[0], outputs[1]) and np.allclose(outputs[1], outputs[2])}")
    
    print("\n💥 Complete God-damned Network Demonstration:")
    print("-" * 50)
    
    # Create and test the complete frustrating network
    god_damned_net = create_god_damned_network()
    
    print("Processing the same input multiple times...")
    results = []
    for i in range(3):
        output = god_damned_net.forward(test_input)
        results.append(output)
        print(f"   Run {i+1}: output range [{np.min(output):.3f}, {np.max(output):.3f}]")
    
    # Check for non-determinism (from NonConvergentLayer)
    diff1 = np.linalg.norm(results[0] - results[1])
    diff2 = np.linalg.norm(results[1] - results[2])
    print(f"   Non-determinism: run1-run2 diff = {diff1:.6f}, run2-run3 diff = {diff2:.6f}")
    
    print("\n🤬 Network Diagnosis:")
    print(f"   ✗ Vanishing gradients: Signal reduced by {((np.linalg.norm(test_input) - np.linalg.norm(results[0])) / np.linalg.norm(test_input) * 100):.1f}%")
    print(f"   ✗ Non-convergent: Output changes between runs")
    print(f"   ✗ Mode collapse: Limited output diversity")
    print(f"   ✗ Overfitting: Memorizes specific inputs")
    print(f"   ✗ Catastrophic forgetting: Forgets previous patterns")
    
    print("\n😅 Mildly Annoying Network (for comparison):")
    print("-" * 45)
    
    annoying_net = create_mildly_annoying_network()
    annoying_output = annoying_net.forward(test_input)
    print(f"   Output range: [{np.min(annoying_output):.3f}, {np.max(annoying_output):.3f}]")
    print("   This one is just annoying enough to be frustrating but not completely broken.")
    
    print("\n" + "=" * 70)
    print("🎭 Conclusion: Neural networks can be beautifully frustrating!")
    print("   These algebraic implementations capture the essence of ML's")
    print("   most annoying behaviors without requiring actual training.")
    print("   Remember: If your neural network isn't occasionally frustrating,")
    print("   you're probably not pushing the boundaries hard enough!")
    print("\n💡 Pro tip: In real life, use batch normalization, proper")
    print("   initialization, and regularization to avoid these issues.")


def main():
    """Run the complete god-damned neural networks demonstration."""
    demonstrate_god_damned_networks()


if __name__ == "__main__":
    main()