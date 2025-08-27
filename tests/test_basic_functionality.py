"""
Simple test script to verify both neural network implementations work correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import numpy as np

# Import our implementations
from fermi_dirac_brain import FermiDiracBrain, create_fermi_dirac_classifier
from maxwell_boltzmann_brain import MaxwellBoltzmannBrain, create_maxwell_boltzmann_classifier


def test_fermi_dirac_brain():
    """Test basic functionality of Fermi-Dirac Brain."""
    print("Testing Fermi-Dirac Brain...")
    
    # Create a simple network
    model = create_fermi_dirac_classifier(
        input_size=10,
        num_classes=2,
        hidden_sizes=[32, 16],
        temperature=1.0
    )
    
    # Test forward pass
    x = torch.randn(5, 10)  # batch_size=5, input_size=10
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test quantum properties
    entropy = model.compute_quantum_entropy()
    fermi_energy = model.get_fermi_energy()
    
    print(f"Quantum entropy: {entropy:.4f}")
    print(f"Fermi energy: {fermi_energy:.4f}")
    
    # Test temperature adjustment
    model.adjust_temperature(0.5)
    new_output = model(x)
    print(f"Output after temperature change: [{new_output.min().item():.3f}, {new_output.max().item():.3f}]")
    
    print("Fermi-Dirac Brain test passed!\n")
    return True


def test_maxwell_boltzmann_brain():
    """Test basic functionality of Maxwell-Boltzmann Brain."""
    print("Testing Maxwell-Boltzmann Brain...")
    
    # Create a simple network
    model = create_maxwell_boltzmann_classifier(
        input_size=10,
        num_classes=2,
        hidden_sizes=[32, 16],
        temperature=2.0
    )
    
    # Test forward pass
    x = torch.randn(5, 10)  # batch_size=5, input_size=10
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test thermodynamic properties
    thermal_energy = model.compute_thermal_energy()
    partition_function = model.compute_partition_function()
    
    print(f"Thermal energy: {thermal_energy:.4f}")
    print(f"Partition function: {partition_function:.4f}")
    
    # Test thermal annealing
    temp_before = model.temperature
    new_temp = model.thermal_anneal(50, 100, initial_temp=2.0, final_temp=0.2)
    print(f"Temperature before annealing: {temp_before:.3f}")
    print(f"Temperature after annealing: {new_temp:.3f}")
    
    print("Maxwell-Boltzmann Brain test passed!\n")
    return True


def test_training_compatibility():
    """Test that both networks can be trained with standard PyTorch training loops."""
    print("Testing training compatibility...")
    
    # Generate synthetic data
    batch_size = 32
    input_size = 8
    X = torch.randn(batch_size, input_size)
    y = torch.randint(0, 2, (batch_size,))
    
    # Test Fermi-Dirac Brain
    fd_model = create_fermi_dirac_classifier(input_size, 2, [16], temperature=1.0)
    fd_optimizer = torch.optim.Adam(fd_model.parameters(), lr=0.01)
    fd_criterion = nn.CrossEntropyLoss()
    
    fd_model.train()
    fd_optimizer.zero_grad()
    fd_output = fd_model(X)
    fd_loss = fd_criterion(fd_output, y)
    fd_loss.backward()
    fd_optimizer.step()
    
    print(f"Fermi-Dirac training loss: {fd_loss.item():.4f}")
    
    # Test Maxwell-Boltzmann Brain
    mb_model = create_maxwell_boltzmann_classifier(input_size, 2, [16], temperature=1.0)
    mb_optimizer = torch.optim.Adam(mb_model.parameters(), lr=0.01)
    mb_criterion = nn.CrossEntropyLoss()
    
    mb_model.train()
    mb_optimizer.zero_grad()
    mb_output = mb_model(X)
    mb_loss = mb_criterion(mb_output, y)
    mb_loss.backward()
    mb_optimizer.step()
    
    print(f"Maxwell-Boltzmann training loss: {mb_loss.item():.4f}")
    print("Training compatibility test passed!\n")
    return True


def test_activation_functions():
    """Test the statistical activation functions directly."""
    print("Testing activation functions...")
    
    # Import activation functions
    from fermi_dirac_brain import FermiDiracActivation
    from maxwell_boltzmann_brain import MaxwellBoltzmannActivation
    
    # Test Fermi-Dirac activation
    fd_activation = FermiDiracActivation(temperature=1.0, chemical_potential=0.0)
    x = torch.linspace(-5, 5, 100)
    fd_output = fd_activation(x)
    
    print(f"Fermi-Dirac activation range: [{fd_output.min().item():.3f}, {fd_output.max().item():.3f}]")
    print(f"Fermi-Dirac at x=0: {fd_activation(torch.tensor(0.0)).item():.3f}")
    
    # Test Maxwell-Boltzmann activation
    mb_activation = MaxwellBoltzmannActivation(temperature=1.0, energy_shift=0.0)
    mb_output = mb_activation(x)
    
    print(f"Maxwell-Boltzmann activation range: [{mb_output.min().item():.3f}, {mb_output.max().item():.3f}]")
    print(f"Maxwell-Boltzmann at x=0: {mb_activation(torch.tensor(0.0)).item():.3f}")
    
    print("Activation function test passed!\n")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("=== Statistical Physics Neural Networks Test Suite ===\n")
    
    tests = [
        test_fermi_dirac_brain,
        test_maxwell_boltzmann_brain,
        test_training_compatibility,
        test_activation_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
        except Exception as e:
            print(f"Test {test.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("✅ All tests passed! The implementations are working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)