"""
Geometric Algebra (Clifford Algebra) Neural Network

This example demonstrates neural networks using geometric algebra operations
for processing geometric and spatial data.
"""

import numpy as np
from typing import List, Tuple, Dict, Union
import math


class GeometricAlgebraNetwork:
    """
    Neural network based on geometric algebra (Clifford algebra) operations.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, signature: str = "euclidean"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.signature = signature
        
        # Initialize geometric algebra basis
        self.basis_elements = self._generate_basis_elements()
        self.metric = self._generate_metric()
        
        # Generate transformation coefficients
        self.coefficients = self._generate_ga_coefficients()
        
    def _generate_metric(self) -> np.ndarray:
        """Generate metric tensor for the geometric algebra."""
        if self.signature == "euclidean":
            return np.eye(self.input_dim)
        elif self.signature == "minkowski":
            metric = np.eye(self.input_dim)
            metric[0, 0] = -1  # Time component with opposite signature
            return metric
        elif self.signature == "conformal":
            # Conformal geometric algebra Cl(n+1,1)
            metric = np.eye(self.input_dim + 2)
            metric[-1, -1] = -1  # One negative signature
            return metric
        else:
            return np.eye(self.input_dim)
    
    def _generate_basis_elements(self) -> List[Tuple[str, np.ndarray]]:
        """Generate basis elements for geometric algebra."""
        basis = []
        
        # Scalar (grade 0)
        scalar_basis = np.zeros(2**self.input_dim)
        scalar_basis[0] = 1.0
        basis.append(("scalar", scalar_basis))
        
        # Vector basis elements (grade 1)
        for i in range(self.input_dim):
            vector_basis = np.zeros(2**self.input_dim)
            vector_basis[2**i] = 1.0
            basis.append((f"e{i+1}", vector_basis))
        
        # Bivector basis elements (grade 2)
        for i in range(self.input_dim):
            for j in range(i+1, self.input_dim):
                bivector_basis = np.zeros(2**self.input_dim)
                bivector_basis[2**i + 2**j] = 1.0
                basis.append((f"e{i+1}e{j+1}", bivector_basis))
        
        # Higher grade elements for small dimensions
        if self.input_dim <= 4:
            # Trivectors (grade 3)
            for i in range(self.input_dim):
                for j in range(i+1, self.input_dim):
                    for k in range(j+1, self.input_dim):
                        trivector_basis = np.zeros(2**self.input_dim)
                        trivector_basis[2**i + 2**j + 2**k] = 1.0
                        basis.append((f"e{i+1}e{j+1}e{k+1}", trivector_basis))
            
            # Pseudoscalar (highest grade)
            if self.input_dim >= 2:
                pseudo_basis = np.zeros(2**self.input_dim)
                pseudo_basis[-1] = 1.0
                basis.append(("pseudoscalar", pseudo_basis))
        
        return basis
    
    def _generate_ga_coefficients(self) -> Dict[str, np.ndarray]:
        """Generate coefficients for geometric algebra transformations."""
        coeffs = {}
        
        # Constants from geometric algebra theory
        sqrt2 = math.sqrt(2)
        sqrt3 = math.sqrt(3)
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        
        constants = [1.0, 1/sqrt2, 1/sqrt3, 1/phi, phi/3, sqrt2/3, sqrt3/5]
        
        # Input to hidden transformation
        num_basis = len(self.basis_elements)
        coeffs['input_hidden'] = np.zeros((self.hidden_dim, self.input_dim, num_basis))
        
        for i in range(self.hidden_dim):
            for j in range(self.input_dim):
                for k in range(num_basis):
                    const_idx = (i + j + k) % len(constants)
                    coeffs['input_hidden'][i, j, k] = constants[const_idx]
        
        # Hidden to output transformation
        coeffs['hidden_output'] = np.zeros((self.output_dim, self.hidden_dim, num_basis))
        
        for i in range(self.output_dim):
            for j in range(self.hidden_dim):
                for k in range(num_basis):
                    const_idx = (i + j + k + 1) % len(constants)
                    coeffs['hidden_output'][i, j, k] = constants[const_idx]
        
        return coeffs
    
    def geometric_product(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute geometric product of two multivectors."""
        # Simplified geometric product implementation
        # In full implementation, this would use the basis multiplication table
        
        if len(a) != len(b):
            min_len = min(len(a), len(b))
            a, b = a[:min_len], b[:min_len]
        
        # For this implementation, approximate with:
        # ab = a·b + a∧b (dot + wedge products)
        
        # Dot product component (grade reduction)
        dot_product = np.dot(a, b)
        
        # Wedge product component (grade increase) - simplified
        wedge_magnitude = np.linalg.norm(np.outer(a, b) - np.outer(b, a))
        
        # Combine into multivector representation
        result = np.zeros(max(len(a), 2**self.input_dim))
        result[0] = dot_product  # Scalar part
        
        if len(result) > 1:
            result[1] = wedge_magnitude  # Vector part approximation
        
        # Additional components based on input structure
        for i in range(2, min(len(result), len(a) + len(b) - 1)):
            result[i] = (a[i % len(a)] * b[i % len(b)] + 
                        b[i % len(b)] * a[i % len(a)]) / 2
        
        return result[:len(a)]
    
    def outer_product(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute outer (wedge) product magnitude."""
        if len(a) >= 2 and len(b) >= 2:
            # 2D outer product as determinant
            return abs(a[0] * b[1] - a[1] * b[0])
        else:
            # Higher dimensional approximation
            return np.linalg.norm(np.outer(a, b) - np.outer(b, a))
    
    def inner_product(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute inner (dot) product."""
        return np.dot(a, b)
    
    def reverse(self, mv: np.ndarray) -> np.ndarray:
        """Compute reverse of multivector (reverse order of basis elements)."""
        # For bivectors and higher grades, reverse changes sign
        reversed_mv = mv.copy()
        
        # Approximate reversal by alternating signs for higher components
        for i in range(1, len(reversed_mv)):
            grade = bin(i).count('1')  # Grade based on binary representation
            if grade % 4 in [2, 3]:  # Bivectors and trivectors change sign
                reversed_mv[i] *= -1
                
        return reversed_mv
    
    def magnitude(self, mv: np.ndarray) -> float:
        """Compute magnitude of multivector."""
        # Magnitude is sqrt(mv * reverse(mv))
        reversed_mv = self.reverse(mv)
        product = self.geometric_product(mv, reversed_mv)
        return math.sqrt(abs(product[0]))  # Scalar part should be positive
    
    def normalize(self, mv: np.ndarray) -> np.ndarray:
        """Normalize multivector."""
        mag = self.magnitude(mv)
        if mag > 1e-10:
            return mv / mag
        else:
            return mv
    
    def apply_ga_transformation(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Apply geometric algebra transformation."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        batch_size, input_size = x.shape
        output_size = coeffs.shape[0]
        num_basis = len(self.basis_elements)
        
        result = np.zeros((batch_size, output_size))
        
        for batch_idx in range(batch_size):
            for out_idx in range(output_size):
                # Construct multivector from input
                multivector = np.zeros(num_basis)
                
                for in_idx in range(min(input_size, self.input_dim)):
                    for basis_idx in range(num_basis):
                        basis_name, basis_vector = self.basis_elements[basis_idx]
                        coeff = coeffs[out_idx, in_idx, basis_idx]
                        
                        # Ensure basis_vector has correct length
                        basis_component = basis_vector[:num_basis] if len(basis_vector) >= num_basis else np.pad(basis_vector, (0, num_basis - len(basis_vector)))
                        
                        # Weight by input value and coefficient
                        multivector += x[batch_idx, in_idx] * coeff * basis_component
                
                # Apply geometric algebra operations
                # 1. Geometric product with basis elements
                transformed_mv = multivector.copy()
                
                for basis_idx in range(min(3, num_basis)):  # Use first few basis elements
                    _, basis_vector = self.basis_elements[basis_idx]
                    transformed_mv = self.geometric_product(transformed_mv, basis_vector[:num_basis])
                
                # 2. Extract scalar and vector parts
                scalar_part = transformed_mv[0] if len(transformed_mv) > 0 else 0
                vector_magnitude = np.linalg.norm(transformed_mv[1:min(4, len(transformed_mv))])
                
                # 3. Combine into output
                result[batch_idx, out_idx] = scalar_part + vector_magnitude
        
        return result
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through geometric algebra network."""
        # Input to hidden layer
        hidden = self.apply_ga_transformation(x, self.coefficients['input_hidden'])
        
        # Apply nonlinearity (preserve geometric structure)
        hidden = np.tanh(hidden)
        
        # Hidden to output layer
        output = self.apply_ga_transformation(hidden, self.coefficients['hidden_output'])
        
        return output
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Prediction method."""
        return self.forward(x)


def test_geometric_operations():
    """Test basic geometric algebra operations."""
    print("=== Geometric Algebra: Basic Operations Test ===\n")
    
    network = GeometricAlgebraNetwork(input_dim=3, hidden_dim=4, output_dim=2)
    
    # Test vectors
    a = np.array([1, 0, 0])  # e1
    b = np.array([0, 1, 0])  # e2
    c = np.array([1, 1, 0])  # e1 + e2
    
    print("Testing geometric algebra operations:")
    
    # Inner products
    inner_ab = network.inner_product(a, b)
    inner_aa = network.inner_product(a, a)
    print(f"Inner product a·b = {inner_ab:.3f} (should be 0)")
    print(f"Inner product a·a = {inner_aa:.3f} (should be 1)")
    
    # Outer products
    outer_ab = network.outer_product(a, b)
    outer_aa = network.outer_product(a, a)
    print(f"Outer product a∧b magnitude = {outer_ab:.3f} (should be 1)")
    print(f"Outer product a∧a magnitude = {outer_aa:.3f} (should be 0)")
    
    # Geometric products
    geom_ab = network.geometric_product(a, b)
    print(f"Geometric product a*b = {geom_ab[:3]} (first 3 components)")
    
    # Magnitudes
    mag_a = network.magnitude(a)
    mag_c = network.magnitude(c)
    print(f"Magnitude |a| = {mag_a:.3f}")
    print(f"Magnitude |c| = {mag_c:.3f}")
    
    print()


def test_3d_rotation_processing():
    """Test processing of 3D rotational data."""
    print("=== Geometric Algebra: 3D Rotation Processing ===\n")
    
    network = GeometricAlgebraNetwork(input_dim=3, hidden_dim=6, output_dim=4)
    
    # Generate 3D rotation data (axis-angle representation)
    rotation_axes = [
        [1, 0, 0],  # X-axis rotation
        [0, 1, 0],  # Y-axis rotation
        [0, 0, 1],  # Z-axis rotation
        [1, 1, 1],  # Diagonal rotation
        [1, -1, 0], # Mixed rotation
    ]
    
    print("Processing 3D rotation data:")
    
    for i, axis in enumerate(rotation_axes):
        axis = np.array(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)  # Normalize
        
        # Different rotation angles
        angles = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
        
        outputs = []
        for angle in angles:
            # Rotation vector (axis * angle)
            rotation_vector = axis * angle
            output = network.predict(rotation_vector.reshape(1, -1))
            outputs.append(output[0])
        
        outputs = np.array(outputs)
        
        print(f"\nRotation axis {i+1}: {axis}")
        print(f"  Output range: [{np.min(outputs):.3f}, {np.max(outputs):.3f}]")
        print(f"  Output variance: {np.var(outputs, axis=0)}")
        
        # Check for periodic behavior (rotations should have 2π periodicity)
        first_output = outputs[0]  # 0 radians
        last_output = outputs[-1]   # 3π/2 radians, should be similar to π/2
        
        periodicity_error = np.linalg.norm(first_output - outputs[2])  # Compare 0 and π
        print(f"  Periodicity error (0 vs π): {periodicity_error:.6f}")


def test_conformal_geometry():
    """Test conformal geometric algebra for 2D points."""
    print("=== Geometric Algebra: Conformal Geometry ===\n")
    
    # Use conformal signature for 2D conformal GA
    network = GeometricAlgebraNetwork(input_dim=2, hidden_dim=8, output_dim=4, signature="conformal")
    
    # Test geometric primitives
    geometric_objects = {
        "Point": np.array([1, 1]),
        "Origin": np.array([0, 0]),
        "Unit_X": np.array([1, 0]),
        "Unit_Y": np.array([0, 1]),
        "Diagonal": np.array([1, 1]) / np.sqrt(2)
    }
    
    print("Processing geometric objects in conformal space:")
    
    object_features = {}
    for obj_name, point in geometric_objects.items():
        # Convert to conformal representation
        # In conformal GA: P = point + 0.5*|point|²*e∞ + e₀
        point_squared = np.dot(point, point)
        conformal_point = np.concatenate([point, [0.5 * point_squared, 1]])
        
        # Process through network
        output = network.predict(conformal_point.reshape(1, -1))
        object_features[obj_name] = output[0]
        
        print(f"{obj_name:>10}: {point} → output: {output[0]}")
    
    # Analyze relationships between objects
    print("\nAnalyzing geometric relationships:")
    
    origin_features = object_features["Origin"]
    for obj_name, features in object_features.items():
        if obj_name != "Origin":
            distance = np.linalg.norm(features - origin_features)
            print(f"  Feature distance from origin to {obj_name}: {distance:.4f}")


def test_bivector_operations():
    """Test bivector operations for oriented areas."""
    print("=== Geometric Algebra: Bivector Operations ===\n")
    
    network = GeometricAlgebraNetwork(input_dim=4, hidden_dim=6, output_dim=3)
    
    # Create bivectors representing oriented areas
    bivectors = [
        [1, 0, 1, 0],  # e1∧e3
        [0, 1, 0, 1],  # e2∧e4
        [1, 1, 0, 0],  # e1∧e2
        [0, 0, 1, 1],  # e3∧e4
        [1, 0, 0, 1],  # e1∧e4
    ]
    
    print("Processing bivector data:")
    
    for i, bivector in enumerate(bivectors):
        bv = np.array(bivector, dtype=float)
        output = network.predict(bv.reshape(1, -1))
        
        # Calculate bivector magnitude
        bv_magnitude = np.linalg.norm(bv)
        
        print(f"Bivector {i+1}: {bv}")
        print(f"  Magnitude: {bv_magnitude:.3f}")
        print(f"  Network output: {output[0]}")
        print(f"  Output magnitude: {np.linalg.norm(output[0]):.3f}")
        print()


def test_multivector_algebra():
    """Test general multivector operations."""
    print("=== Geometric Algebra: Multivector Operations ===\n")
    
    network = GeometricAlgebraNetwork(input_dim=3, hidden_dim=5, output_dim=2)
    
    # Create multivectors with different grade components
    multivectors = [
        [1, 0, 0],     # Pure vector e1
        [0, 1, 0],     # Pure vector e2
        [0, 0, 1],     # Pure vector e3
        [1, 1, 0],     # e1 + e2
        [1, 1, 1],     # e1 + e2 + e3
        [2, -1, 0.5],  # 2*e1 - e2 + 0.5*e3
    ]
    
    print("Multivector algebra processing:")
    
    for i, mv in enumerate(multivectors):
        mv_array = np.array(mv, dtype=float)
        
        # Test reverse operation
        reversed_mv = network.reverse(mv_array)
        
        # Test magnitude
        magnitude = network.magnitude(mv_array)
        
        # Test normalization
        normalized_mv = network.normalize(mv_array)
        
        # Network processing
        output = network.predict(mv_array.reshape(1, -1))
        
        print(f"\nMultivector {i+1}: {mv}")
        print(f"  Reversed: {reversed_mv}")
        print(f"  Magnitude: {magnitude:.4f}")
        print(f"  Normalized: {normalized_mv}")
        print(f"  Network output: {output[0]}")


if __name__ == "__main__":
    print("Geometric Algebra Neural Network Demo\n")
    print("="*60)
    
    # Run tests
    test_geometric_operations()
    test_3d_rotation_processing()
    test_conformal_geometry()
    test_bivector_operations()
    test_multivector_algebra()
    
    print("\n" + "="*60)
    print("Geometric algebra demo completed successfully!")