"""
Polynomial-based Algebraic Neural Network

This example demonstrates a neural network that uses polynomial transformations
with coefficients derived from algebraic number theory.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import math


class PolynomialAlgebraicNetwork:
    """
    Neural network using polynomial basis functions with algebraic coefficients.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, max_degree: int = 3):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_degree = max_degree
        
        # Generate polynomial coefficients using algebraic numbers
        self.coefficients = self._generate_algebraic_coefficients()
        
    def _generate_algebraic_coefficients(self) -> dict:
        """Generate coefficients using famous algebraic constants."""
        coeffs = {}
        
        # Golden ratio and related algebraic numbers
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        phi_conjugate = (1 - math.sqrt(5)) / 2
        
        # Silver ratio
        silver = 1 + math.sqrt(2)
        
        # Euler's number approximation using continued fractions
        e_approx = 2.718281828
        
        # Pi approximation
        pi_approx = math.pi
        
        algebraic_constants = [1, phi, phi_conjugate, silver, e_approx, pi_approx]
        
        # Generate coefficients for input to hidden transformation
        coeffs['input_hidden'] = np.zeros((self.hidden_dim, self.input_dim, self.max_degree + 1))
        for i in range(self.hidden_dim):
            for j in range(self.input_dim):
                for k in range(self.max_degree + 1):
                    # Use algebraic constants in a systematic way
                    const_idx = (i + j + k) % len(algebraic_constants)
                    base_coeff = algebraic_constants[const_idx]
                    
                    # Scale by factorial to maintain stability
                    coeffs['input_hidden'][i, j, k] = base_coeff / math.factorial(k + 1)
        
        # Generate coefficients for hidden to output transformation
        coeffs['hidden_output'] = np.zeros((self.output_dim, self.hidden_dim, self.max_degree + 1))
        for i in range(self.output_dim):
            for j in range(self.hidden_dim):
                for k in range(self.max_degree + 1):
                    const_idx = (i + j + k + 1) % len(algebraic_constants)
                    base_coeff = algebraic_constants[const_idx]
                    coeffs['hidden_output'][i, j, k] = base_coeff / math.factorial(k + 1)
        
        return coeffs
    
    def _polynomial_activation(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Apply polynomial activation with given coefficients."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        batch_size, input_size = x.shape
        output_size = coeffs.shape[0]
        
        result = np.zeros((batch_size, output_size))
        
        for i in range(output_size):
            for j in range(input_size):
                for degree in range(self.max_degree + 1):
                    if degree == 0:
                        poly_term = coeffs[i, j, degree]
                    else:
                        poly_term = coeffs[i, j, degree] * (x[:, j] ** degree)
                    result[:, i] += poly_term
                    
        return result
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the polynomial network."""
        # Input to hidden layer
        hidden = self._polynomial_activation(x, self.coefficients['input_hidden'])
        
        # Apply hyperbolic tangent for stability
        hidden = np.tanh(hidden)
        
        # Hidden to output layer
        output = self._polynomial_activation(hidden, self.coefficients['hidden_output'])
        
        return output
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Prediction method."""
        return self.forward(x)


def test_function_approximation():
    """Test the polynomial network on function approximation tasks."""
    print("=== Polynomial Network Function Approximation ===\n")
    
    # Create network
    network = PolynomialAlgebraicNetwork(input_dim=1, hidden_dim=5, output_dim=1, max_degree=3)
    
    # Test on various mathematical functions
    test_functions = [
        ("Sine", lambda x: np.sin(2 * np.pi * x)),
        ("Cosine", lambda x: np.cos(2 * np.pi * x)),
        ("Quadratic", lambda x: x**2 - 0.5*x + 0.1),
        ("Cubic", lambda x: x**3 - x**2 + 0.5*x),
        ("Exponential", lambda x: np.exp(-x**2))
    ]
    
    x_test = np.linspace(-1, 1, 50).reshape(-1, 1)
    
    results = {}
    for func_name, func in test_functions:
        y_true = func(x_test.flatten())
        y_pred = network.predict(x_test).flatten()
        
        # Calculate approximation error
        mse = np.mean((y_true - y_pred)**2)
        mae = np.mean(np.abs(y_true - y_pred))
        
        results[func_name] = {
            'mse': mse,
            'mae': mae,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        print(f"{func_name}:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print()
    
    return results, x_test


def test_pattern_recognition():
    """Test polynomial network on 2D pattern recognition."""
    print("=== Polynomial Network Pattern Recognition ===\n")
    
    # Create 2D network
    network = PolynomialAlgebraicNetwork(input_dim=2, hidden_dim=8, output_dim=3, max_degree=2)
    
    # Generate test patterns
    def generate_circle_points(n_points=20, radius=0.8):
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    
    def generate_square_points(n_points=20, side=1.0):
        points_per_side = n_points // 4
        side_points = []
        
        # Bottom side
        x = np.linspace(-side/2, side/2, points_per_side)
        y = np.full(points_per_side, -side/2)
        side_points.extend(zip(x, y))
        
        # Right side
        x = np.full(points_per_side, side/2)
        y = np.linspace(-side/2, side/2, points_per_side)
        side_points.extend(zip(x, y))
        
        # Top side
        x = np.linspace(side/2, -side/2, points_per_side)
        y = np.full(points_per_side, side/2)
        side_points.extend(zip(x, y))
        
        # Left side
        x = np.full(points_per_side, -side/2)
        y = np.linspace(side/2, -side/2, points_per_side)
        side_points.extend(zip(x, y))
        
        return np.array(side_points[:n_points])
    
    def generate_triangle_points(n_points=18, size=0.8):
        angles = np.array([0, 2*np.pi/3, 4*np.pi/3])
        vertices = size * np.column_stack([np.cos(angles), np.sin(angles)])
        
        points = []
        points_per_edge = n_points // 3
        
        for i in range(3):
            start = vertices[i]
            end = vertices[(i + 1) % 3]
            edge_points = np.linspace(start, end, points_per_edge, endpoint=False)
            points.extend(edge_points)
        
        return np.array(points[:n_points])
    
    # Generate patterns
    circles = generate_circle_points()
    squares = generate_square_points()
    triangles = generate_triangle_points()
    
    # Process with network
    circle_outputs = network.predict(circles)
    square_outputs = network.predict(squares)
    triangle_outputs = network.predict(triangles)
    
    # Analyze outputs
    print("Circle pattern analysis:")
    print(f"  Mean output: {np.mean(circle_outputs, axis=0)}")
    print(f"  Std output: {np.std(circle_outputs, axis=0)}")
    
    print("\nSquare pattern analysis:")
    print(f"  Mean output: {np.mean(square_outputs, axis=0)}")
    print(f"  Std output: {np.std(square_outputs, axis=0)}")
    
    print("\nTriangle pattern analysis:")
    print(f"  Mean output: {np.mean(triangle_outputs, axis=0)}")
    print(f"  Std output: {np.std(triangle_outputs, axis=0)}")
    
    return {
        'circles': (circles, circle_outputs),
        'squares': (squares, square_outputs),
        'triangles': (triangles, triangle_outputs)
    }


def demonstrate_coefficient_properties():
    """Demonstrate properties of the algebraic coefficients."""
    print("=== Algebraic Coefficient Properties ===\n")
    
    network = PolynomialAlgebraicNetwork(input_dim=3, hidden_dim=4, output_dim=2)
    
    # Analyze coefficient matrices
    input_hidden_coeffs = network.coefficients['input_hidden']
    hidden_output_coeffs = network.coefficients['hidden_output']
    
    print("Input-Hidden Coefficients:")
    print(f"  Shape: {input_hidden_coeffs.shape}")
    print(f"  Min coefficient: {np.min(input_hidden_coeffs):.6f}")
    print(f"  Max coefficient: {np.max(input_hidden_coeffs):.6f}")
    print(f"  Mean coefficient: {np.mean(input_hidden_coeffs):.6f}")
    print(f"  Std coefficient: {np.std(input_hidden_coeffs):.6f}")
    
    print("\nHidden-Output Coefficients:")
    print(f"  Shape: {hidden_output_coeffs.shape}")
    print(f"  Min coefficient: {np.min(hidden_output_coeffs):.6f}")
    print(f"  Max coefficient: {np.max(hidden_output_coeffs):.6f}")
    print(f"  Mean coefficient: {np.mean(hidden_output_coeffs):.6f}")
    print(f"  Std coefficient: {np.std(hidden_output_coeffs):.6f}")
    
    # Test stability with different input magnitudes
    print("\nStability Analysis:")
    test_inputs = [
        np.array([[0.1, 0.1, 0.1]]),
        np.array([[0.5, 0.5, 0.5]]),
        np.array([[1.0, 1.0, 1.0]]),
        np.array([[2.0, 2.0, 2.0]]),
    ]
    
    for i, test_input in enumerate(test_inputs):
        output = network.predict(test_input)
        magnitude = np.linalg.norm(test_input)
        output_magnitude = np.linalg.norm(output)
        print(f"  Input magnitude {magnitude:.1f} → Output magnitude {output_magnitude:.6f}")


if __name__ == "__main__":
    # Run demonstrations
    print("Polynomial Algebraic Neural Network Demo\n")
    print("="*50)
    
    # Function approximation test
    func_results, x_vals = test_function_approximation()
    
    # Pattern recognition test
    pattern_results = test_pattern_recognition()
    
    # Coefficient analysis
    demonstrate_coefficient_properties()
    
    print("\n" + "="*50)
    print("Demo completed successfully!")