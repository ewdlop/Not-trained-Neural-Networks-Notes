"""
Simple test script to validate all neural network implementations.
"""

import torch
import torch.nn as nn
from neural_networks import (
    RandomFeatureNetwork, EchoStateNetwork, 
    ExtremeLearnlingMachine, LiquidStateMachine,
    create_random_dataset
)


def test_random_feature_network():
    """Test Random Feature Network."""
    print("Testing Random Feature Network...")
    
    # Create network
    net = RandomFeatureNetwork(10, 50, 3)
    
    # Test forward pass
    x = torch.randn(32, 10)
    output = net(x)
    
    assert output.shape == (32, 3), f"Expected shape (32, 3), got {output.shape}"
    
    # Check that hidden layer is not trainable
    hidden_trainable = any(p.requires_grad for p in net.hidden_layer.parameters())
    assert not hidden_trainable, "Hidden layer should not be trainable"
    
    # Check that output layer is trainable
    output_trainable = any(p.requires_grad for p in net.output_layer.parameters())
    assert output_trainable, "Output layer should be trainable"
    
    print("✓ Random Feature Network passed all tests")


def test_echo_state_network():
    """Test Echo State Network."""
    print("Testing Echo State Network...")
    
    # Create network
    net = EchoStateNetwork(5, 30, 2)
    
    # Test forward pass
    net.reset_state(16)
    x = torch.randn(16, 5)
    output = net(x)
    
    assert output.shape == (16, 2), f"Expected shape (16, 2), got {output.shape}"
    
    # Check reservoir weights are not trainable
    reservoir_trainable = net.W_res.requires_grad or net.W_in.requires_grad
    assert not reservoir_trainable, "Reservoir weights should not be trainable"
    
    # Check output weights are trainable
    output_trainable = any(p.requires_grad for p in net.W_out.parameters())
    assert output_trainable, "Output weights should be trainable"
    
    print("✓ Echo State Network passed all tests")


def test_extreme_learning_machine():
    """Test Extreme Learning Machine."""
    print("Testing Extreme Learning Machine...")
    
    # Create network
    net = ExtremeLearnlingMachine(8, 40, 1)
    
    # Test forward pass
    x = torch.randn(20, 8)
    output = net(x)
    
    assert output.shape == (20, 1), f"Expected shape (20, 1), got {output.shape}"
    
    # Test analytical training
    y = torch.randn(20, 1)
    net.fit_analytical(x, y)
    
    # Test prediction after analytical training
    new_output = net(x)
    assert new_output.shape == (20, 1), f"Expected shape (20, 1), got {new_output.shape}"
    
    # Check that input weights are not trainable
    input_trainable = net.input_weights.requires_grad or net.biases.requires_grad
    assert not input_trainable, "Input weights and biases should not be trainable"
    
    print("✓ Extreme Learning Machine passed all tests")


def test_liquid_state_machine():
    """Test Liquid State Machine."""
    print("Testing Liquid State Machine...")
    
    # Create network
    net = LiquidStateMachine(4, 25, 2)
    
    # Test forward pass
    net.reset_state(12)
    x = torch.randn(12, 4)
    output = net(x)
    
    assert output.shape == (12, 2), f"Expected shape (12, 2), got {output.shape}"
    
    # Check liquid weights are not trainable
    liquid_trainable = net.W_in.requires_grad or net.W_liquid.requires_grad
    assert not liquid_trainable, "Liquid weights should not be trainable"
    
    # Check readout is trainable
    readout_trainable = any(p.requires_grad for p in net.readout.parameters())
    assert readout_trainable, "Readout should be trainable"
    
    print("✓ Liquid State Machine passed all tests")


def test_create_random_dataset():
    """Test dataset creation function."""
    print("Testing dataset creation...")
    
    X, y = create_random_dataset(100, 15, 3, 0.1)
    
    assert X.shape == (100, 15), f"Expected X shape (100, 15), got {X.shape}"
    assert y.shape == (100, 3), f"Expected y shape (100, 3), got {y.shape}"
    
    print("✓ Dataset creation passed all tests")


def test_training_comparison():
    """Test that networks can be trained and produce reasonable outputs."""
    print("Testing training capabilities...")
    
    # Create common dataset
    X_train, y_train = create_random_dataset(200, 10, 1, 0.05)
    X_test, y_test = create_random_dataset(50, 10, 1, 0.05)
    
    networks = {
        'RFN': RandomFeatureNetwork(10, 80, 1),
        'ESN': EchoStateNetwork(10, 40, 1),
        'ELM': ExtremeLearnlingMachine(10, 60, 1),
        'LSM': LiquidStateMachine(10, 30, 1)
    }
    
    results = {}
    
    for name, net in networks.items():
        if name == 'ELM':
            # Analytical training
            net.fit_analytical(X_train, y_train)
        else:
            # Gradient-based training of output layer only
            optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=0.01)
            criterion = nn.MSELoss()
            
            for epoch in range(50):
                if name in ['ESN', 'LSM']:
                    net.reset_state(X_train.size(0))
                
                optimizer.zero_grad()
                outputs = net(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
        
        # Test
        net.eval()
        with torch.no_grad():
            if name in ['ESN', 'LSM']:
                net.reset_state(X_test.size(0))
            
            test_outputs = net(X_test)
            test_mse = nn.MSELoss()(test_outputs, y_test)
            results[name] = test_mse.item()
    
    # Check that all networks produce reasonable results
    for name, mse in results.items():
        assert mse < 100.0, f"{name} produced unreasonably high MSE: {mse}"
        assert not torch.isnan(torch.tensor(mse)), f"{name} produced NaN MSE"
        
    print(f"✓ Training test passed. MSE results: {results}")


def run_all_tests():
    """Run all tests."""
    print("Running all tests for Neural Networks that don't need training")
    print("=" * 70)
    
    try:
        test_random_feature_network()
        test_echo_state_network()
        test_extreme_learning_machine()
        test_liquid_state_machine()
        test_create_random_dataset()
        test_training_comparison()
        
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("Your neural networks are ready to use!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()