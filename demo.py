#!/usr/bin/env python3
"""
Demo script for Type Theory Neural Network

This script demonstrates the type theory neural network implementation
with various examples and use cases.
"""

import torch
import numpy as np
from type_theory_nn import (
    TypeTheoryNeuralNetwork, 
    create_classification_network,
    create_regression_network,
    TypedTensor,
    TypedLinear,
    TypedActivation
)


def demo_basic_usage():
    """Demonstrate basic usage of the type theory neural network."""
    print("=" * 60)
    print("DEMO 1: Basic Type Theory Neural Network Usage")
    print("=" * 60)
    
    # Create a simple network
    net = TypeTheoryNeuralNetwork(
        layer_sizes=[4, 8, 6, 3],
        activations=['relu', 'tanh', 'softmax'],
        input_type="InputFeatures",
        output_type="ClassProbabilities"
    )
    
    print("Network Architecture:")
    print(f"Type signature: {net.get_type_signature()}")
    
    # Test with sample data
    batch_size = 5
    sample_input = torch.randn(batch_size, 4)
    
    print(f"\nInput tensor shape: {sample_input.shape}")
    print(f"Input tensor type: InputFeatures")
    
    # Forward pass with type tracking
    output, type_trace = net(sample_input, "InputFeatures")
    
    print(f"\nOutput tensor shape: {output.shape}")
    print(f"Type evolution through network:")
    for i, type_name in enumerate(type_trace):
        print(f"  Step {i}: {type_name}")
    
    print(f"\nOutput probabilities (first sample):")
    print(f"  {output[0].detach().numpy()}")
    print(f"  Sum: {output[0].sum().item():.6f} (should be ~1.0 for probabilities)")


def demo_classification():
    """Demonstrate classification network with type checking."""
    print("\n" + "=" * 60)
    print("DEMO 2: Classification Network")
    print("=" * 60)
    
    # Create classification network
    net = create_classification_network(
        input_size=10,
        hidden_sizes=[20, 15],
        num_classes=5
    )
    
    print("Classification Network:")
    print(f"Type signature: {net.get_type_signature()}")
    
    # Generate fake classification data
    n_samples = 8
    X = torch.randn(n_samples, 10)
    
    print(f"\nInput: {n_samples} samples, 10 features each")
    
    # Forward pass
    output, type_trace = net(X, "FeatureVector")
    
    print(f"Output shape: {output.shape}")
    print(f"Type trace: {' -> '.join(type_trace)}")
    
    # Show predictions
    predicted_classes = torch.argmax(output, dim=1)
    print(f"\nPredicted classes: {predicted_classes.tolist()}")
    print(f"Confidence scores (max probabilities): {torch.max(output, dim=1)[0].tolist()}")


def demo_regression():
    """Demonstrate regression network."""
    print("\n" + "=" * 60)
    print("DEMO 3: Regression Network")
    print("=" * 60)
    
    # Create regression network
    net = create_regression_network(
        input_size=5,
        hidden_sizes=[10, 8],
        output_size=2
    )
    
    print("Regression Network:")
    print(f"Type signature: {net.get_type_signature()}")
    
    # Generate fake regression data
    n_samples = 6
    X = torch.randn(n_samples, 5)
    
    print(f"\nInput: {n_samples} samples, 5 features each")
    
    # Forward pass
    output, type_trace = net(X, "FeatureVector")
    
    print(f"Output shape: {output.shape}")
    print(f"Type trace: {' -> '.join(type_trace)}")
    
    print(f"\nRegression outputs:")
    for i, pred in enumerate(output):
        print(f"  Sample {i+1}: [{pred[0].item():.4f}, {pred[1].item():.4f}]")


def demo_typed_components():
    """Demonstrate individual typed components."""
    print("\n" + "=" * 60)
    print("DEMO 4: Individual Typed Components")
    print("=" * 60)
    
    # Test TypedTensor
    print("TypedTensor Example:")
    tensor = torch.randn(3, 4)
    typed_tensor = TypedTensor(tensor, "FeatureMatrix")
    print(f"  {typed_tensor}")
    
    # Test TypedLinear
    print("\nTypedLinear Layer:")
    linear_layer = TypedLinear(4, 6, "FeatureMatrix", "HiddenRepresentation")
    print(f"  Input type: {linear_layer.get_input_type()}")
    print(f"  Output type: {linear_layer.get_output_type()}")
    
    output = linear_layer(typed_tensor)
    print(f"  Output: {output}")
    
    # Test TypedActivation
    print("\nTypedActivation Layer:")
    activation_layer = TypedActivation('relu', "ReLU_HiddenRepresentation")
    activated_output = activation_layer(output)
    print(f"  Activated output: {activated_output}")
    
    # Test type checking error
    print("\nType Checking Error Example:")
    try:
        wrong_typed_tensor = TypedTensor(tensor, "WrongType")
        linear_layer(wrong_typed_tensor)
    except TypeError as e:
        print(f"  Caught expected error: {e}")


def demo_type_compatibility():
    """Demonstrate type compatibility checking."""
    print("\n" + "=" * 60)
    print("DEMO 5: Type Compatibility Checking")
    print("=" * 60)
    
    net = create_classification_network(input_size=4, hidden_sizes=[8], num_classes=3)
    
    # Test different tensor types
    real_tensor = torch.randn(2, 4)
    int_tensor = torch.randint(0, 10, (2, 4))
    prob_tensor = torch.softmax(torch.randn(2, 3), dim=1)
    
    print("Type Compatibility Tests:")
    print(f"  Real tensor with 'Real' type: {net.check_type_compatibility(real_tensor, 'Real')}")
    print(f"  Integer tensor with 'Integer' type: {net.check_type_compatibility(int_tensor, 'Integer')}")
    print(f"  Integer tensor with 'Real' type: {net.check_type_compatibility(int_tensor, 'Real')}")
    print(f"  Probability tensor with 'Probability' type: {net.check_type_compatibility(prob_tensor, 'Probability')}")
    
    print(f"\nProbability tensor sums: {prob_tensor.sum(dim=1).tolist()}")


if __name__ == "__main__":
    print("Type Theory Neural Network Demo")
    print("This demonstrates a PyTorch neural network with type theory concepts")
    print()
    
    try:
        demo_basic_usage()
        demo_classification()
        demo_regression()
        demo_typed_components()
        demo_type_compatibility()
        
        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()