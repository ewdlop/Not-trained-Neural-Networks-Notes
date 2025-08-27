#!/usr/bin/env python3
"""
Simple test script for Type Theory Neural Network

This script runs basic tests to ensure the implementation works correctly.
"""

import torch
import sys
from type_theory_nn import (
    TypeTheoryNeuralNetwork,
    create_classification_network,
    create_regression_network,
    TypedTensor,
    TypedLinear,
    TypedActivation
)


def test_typed_tensor():
    """Test TypedTensor functionality."""
    print("Testing TypedTensor...")
    tensor = torch.randn(2, 3)
    typed_tensor = TypedTensor(tensor, "TestType")
    
    assert typed_tensor.tensor.shape == torch.Size([2, 3])
    assert typed_tensor.type_info == "TestType"
    print("✓ TypedTensor tests passed")


def test_typed_linear():
    """Test TypedLinear layer."""
    print("Testing TypedLinear...")
    layer = TypedLinear(3, 5, "InputType", "OutputType")
    
    input_tensor = TypedTensor(torch.randn(2, 3), "InputType")
    output = layer(input_tensor)
    
    assert output.tensor.shape == torch.Size([2, 5])
    assert output.type_info == "OutputType"
    
    # Test type checking
    try:
        wrong_input = TypedTensor(torch.randn(2, 3), "WrongType")
        layer(wrong_input)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass  # Expected
    
    print("✓ TypedLinear tests passed")


def test_typed_activation():
    """Test TypedActivation layer."""
    print("Testing TypedActivation...")
    
    # Test ReLU
    relu_layer = TypedActivation('relu', "ReLU_Output")
    input_tensor = TypedTensor(torch.randn(2, 3), "InputType")
    output = relu_layer(input_tensor)
    
    assert output.type_info == "ReLU_Output"
    assert torch.all(output.tensor >= 0)  # ReLU should be non-negative
    
    # Test Softmax
    softmax_layer = TypedActivation('softmax', "Probability")
    output = softmax_layer(input_tensor)
    
    assert output.type_info == "Probability"
    assert torch.allclose(output.tensor.sum(dim=-1), torch.ones(2))  # Should sum to 1
    
    print("✓ TypedActivation tests passed")


def test_type_theory_network():
    """Test the main TypeTheoryNeuralNetwork."""
    print("Testing TypeTheoryNeuralNetwork...")
    
    net = TypeTheoryNeuralNetwork(
        layer_sizes=[4, 6, 3],
        activations=['relu', 'softmax'],
        input_type="Input",
        output_type="Output"
    )
    
    # Test type signature
    signature = net.get_type_signature()
    assert signature['input_type'] == "Input"
    assert signature['output_type'] == "Probability"
    
    # Test forward pass
    input_tensor = torch.randn(5, 4)
    output, type_trace = net(input_tensor, "Input")
    
    assert output.shape == torch.Size([5, 3])
    assert len(type_trace) == 5  # Input + 2 layers (linear+activation each)
    assert type_trace[0] == "Input"
    assert type_trace[-1] == "Probability"
    
    # Test softmax output sums to 1
    assert torch.allclose(output.sum(dim=-1), torch.ones(5))
    
    print("✓ TypeTheoryNeuralNetwork tests passed")


def test_classification_network():
    """Test classification network creation."""
    print("Testing classification network...")
    
    net = create_classification_network(
        input_size=10,
        hidden_sizes=[8, 6],
        num_classes=4
    )
    
    input_tensor = torch.randn(3, 10)
    output, type_trace = net(input_tensor, "FeatureVector")
    
    assert output.shape == torch.Size([3, 4])
    assert type_trace[0] == "FeatureVector"
    assert type_trace[-1] == "Probability"
    
    print("✓ Classification network tests passed")


def test_regression_network():
    """Test regression network creation."""
    print("Testing regression network...")
    
    net = create_regression_network(
        input_size=5,
        hidden_sizes=[8],
        output_size=2
    )
    
    input_tensor = torch.randn(4, 5)
    output, type_trace = net(input_tensor, "FeatureVector")
    
    assert output.shape == torch.Size([4, 2])
    assert type_trace[0] == "FeatureVector"
    assert "Tanh" in type_trace[-1]  # Should end with tanh activation
    
    # Tanh output should be in [-1, 1]
    assert torch.all(output >= -1)
    assert torch.all(output <= 1)
    
    print("✓ Regression network tests passed")


def test_type_compatibility():
    """Test type compatibility checking."""
    print("Testing type compatibility...")
    
    net = create_classification_network(input_size=4, hidden_sizes=[6], num_classes=3)
    
    real_tensor = torch.randn(2, 4)
    int_tensor = torch.randint(0, 10, (2, 4))
    prob_tensor = torch.softmax(torch.randn(2, 3), dim=1)
    
    assert net.check_type_compatibility(real_tensor, 'Real') == True
    assert net.check_type_compatibility(int_tensor, 'Integer') == True
    assert net.check_type_compatibility(int_tensor, 'Real') == False
    assert net.check_type_compatibility(prob_tensor, 'Probability') == True
    
    print("✓ Type compatibility tests passed")


def run_all_tests():
    """Run all tests."""
    print("Running Type Theory Neural Network Tests...")
    print("=" * 50)
    
    try:
        test_typed_tensor()
        test_typed_linear()
        test_typed_activation()
        test_type_theory_network()
        test_classification_network()
        test_regression_network()
        test_type_compatibility()
        
        print("=" * 50)
        print("🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)