"""
Simple tests for the Super-Ego-Id Neural Network implementation.

These tests verify that the network components work correctly and 
that the psychological modeling produces expected behaviors.
"""

import torch
from super_ego_id_network import SuperEgoIdNetwork, IdNetwork, EgoNetwork, SuperEgoNetwork


def test_individual_components():
    """Test that each psychological component can be instantiated and run."""
    input_dim, hidden_dim, output_dim = 10, 32, 8
    batch_size = 16
    
    # Create test input
    x = torch.randn(batch_size, input_dim)
    
    # Test Id Network
    id_net = IdNetwork(input_dim, hidden_dim, output_dim)
    id_output = id_net(x)
    assert id_output.shape == (batch_size, output_dim)
    assert torch.max(torch.abs(id_output)) > 1.0  # Should have high activation
    
    # Test Ego Network  
    ego_net = EgoNetwork(input_dim, hidden_dim, output_dim)
    ego_output = ego_net(x)
    assert ego_output.shape == (batch_size, output_dim)
    assert torch.max(torch.abs(ego_output)) <= 1.1  # Should be more constrained
    
    # Test Super-ego Network
    superego_net = SuperEgoNetwork(input_dim, hidden_dim, output_dim)
    superego_output = superego_net(x)
    assert superego_output.shape == (batch_size, output_dim)
    assert torch.max(torch.abs(superego_output)) <= 0.6  # Should be most constrained
    
    print("✓ All individual components work correctly")


def test_full_network():
    """Test the complete Super-Ego-Id network."""
    input_dim, hidden_dim, output_dim = 15, 48, 12
    batch_size = 8
    
    # Create network and test input
    network = SuperEgoIdNetwork(input_dim, hidden_dim, output_dim)
    x = torch.randn(batch_size, input_dim)
    
    # Test forward pass
    output, components = network(x)
    
    # Check output shapes
    assert output.shape == (batch_size, output_dim)
    assert 'id' in components
    assert 'ego' in components
    assert 'superego' in components
    assert 'attention_weights' in components
    
    # Check component outputs
    assert components['id'].shape == (batch_size, output_dim)
    assert components['ego'].shape == (batch_size, output_dim)
    assert components['superego'].shape == (batch_size, output_dim)
    
    print("✓ Full network forward pass works correctly")


def test_component_analysis():
    """Test the component analysis functionality."""
    input_dim, hidden_dim, output_dim = 20, 64, 16
    batch_size = 10
    
    network = SuperEgoIdNetwork(input_dim, hidden_dim, output_dim)
    x = torch.randn(batch_size, input_dim)
    
    # Test component analysis
    analysis = network.get_component_analysis(x)
    
    # Check analysis keys
    required_keys = ['id_magnitude', 'ego_magnitude', 'superego_magnitude',
                    'id_dominance', 'ego_dominance', 'superego_dominance',
                    'attention_weights', 'final_output', 'component_outputs']
    
    for key in required_keys:
        assert key in analysis, f"Missing key: {key}"
    
    # Check dominance values sum to 1
    total_dominance = (analysis['id_dominance'] + 
                      analysis['ego_dominance'] + 
                      analysis['superego_dominance'])
    assert abs(total_dominance - 1.0) < 1e-6, f"Dominance values don't sum to 1: {total_dominance}"
    
    # Check magnitudes are positive
    assert analysis['id_magnitude'] >= 0
    assert analysis['ego_magnitude'] >= 0  
    assert analysis['superego_magnitude'] >= 0
    
    print("✓ Component analysis works correctly")


def test_gradient_flow():
    """Test that gradients flow properly through the network."""
    input_dim, hidden_dim, output_dim = 10, 32, 8
    batch_size = 4
    
    network = SuperEgoIdNetwork(input_dim, hidden_dim, output_dim)
    x = torch.randn(batch_size, input_dim, requires_grad=True)
    
    # Forward pass
    output, _ = network(x)
    loss = torch.sum(output)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None
    assert torch.sum(torch.abs(x.grad)) > 0
    
    # Check that network parameters have gradients
    for name, param in network.named_parameters():
        assert param.grad is not None, f"No gradient for parameter: {name}"
        assert torch.sum(torch.abs(param.grad)) > 0, f"Zero gradient for parameter: {name}"
    
    print("✓ Gradient flow works correctly")


def test_psychological_behavior():
    """Test that the network exhibits expected psychological behaviors."""
    input_dim, hidden_dim, output_dim = 20, 64, 8
    
    network = SuperEgoIdNetwork(input_dim, hidden_dim, output_dim)
    
    # Create different types of inputs to see if components respond differently
    batch_size = 32
    
    # "Impulsive" input (high positive values - should trigger id more)
    impulsive_input = torch.randn(batch_size, input_dim) + 2.0
    
    # "Constrained" input (values around zero - should trigger ego more)  
    balanced_input = torch.randn(batch_size, input_dim) * 0.1
    
    # "Restrictive" input (negative values - might trigger super-ego more)
    restrictive_input = torch.randn(batch_size, input_dim) - 1.0
    
    # Analyze responses
    impulsive_analysis = network.get_component_analysis(impulsive_input)
    balanced_analysis = network.get_component_analysis(balanced_input)
    restrictive_analysis = network.get_component_analysis(restrictive_input)
    
    print(f"Impulsive scenario - Id: {impulsive_analysis['id_dominance']:.3f}, "
          f"Ego: {impulsive_analysis['ego_dominance']:.3f}, "
          f"Super-ego: {impulsive_analysis['superego_dominance']:.3f}")
    
    print(f"Balanced scenario - Id: {balanced_analysis['id_dominance']:.3f}, "
          f"Ego: {balanced_analysis['ego_dominance']:.3f}, "
          f"Super-ego: {balanced_analysis['superego_dominance']:.3f}")
    
    print(f"Restrictive scenario - Id: {restrictive_analysis['id_dominance']:.3f}, "
          f"Ego: {restrictive_analysis['ego_dominance']:.3f}, "
          f"Super-ego: {restrictive_analysis['superego_dominance']:.3f}")
    
    print("✓ Psychological behavior analysis completed")


def run_all_tests():
    """Run all tests."""
    print("Running Super-Ego-Id Neural Network Tests")
    print("=" * 50)
    
    try:
        test_individual_components()
        test_full_network()
        test_component_analysis()
        test_gradient_flow()
        test_psychological_behavior()
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()