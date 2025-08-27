"""
Group Theory-based Algebraic Neural Network

This example demonstrates neural networks that use group theory operations
for data transformations, particularly focusing on symmetry groups.
"""

import numpy as np
from typing import List, Tuple, Dict
import math


class GroupTheoryNetwork:
    """
    Neural network based on group theory operations and symmetries.
    """
    
    def __init__(self, input_dim: int, group_types: List[str], output_dim: int):
        self.input_dim = input_dim
        self.group_types = group_types
        self.output_dim = output_dim
        
        # Initialize group operations
        self.groups = self._initialize_groups()
        
    def _initialize_groups(self) -> Dict[str, List[np.ndarray]]:
        """Initialize various group operations."""
        groups = {}
        
        for group_type in self.group_types:
            if group_type.startswith("cyclic_"):
                n = int(group_type.split("_")[1])
                groups[group_type] = self._generate_cyclic_group(n)
            elif group_type.startswith("dihedral_"):
                n = int(group_type.split("_")[1])
                groups[group_type] = self._generate_dihedral_group(n)
            elif group_type == "symmetric_3":
                groups[group_type] = self._generate_symmetric_group_3()
            elif group_type == "reflection":
                groups[group_type] = self._generate_reflection_group()
                
        return groups
    
    def _generate_cyclic_group(self, n: int) -> List[np.ndarray]:
        """Generate cyclic group Cn as rotation matrices."""
        group_elements = []
        
        for k in range(n):
            angle = 2 * math.pi * k / n
            
            if self.input_dim == 2:
                # 2D rotation matrix
                rotation = np.array([
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)]
                ])
            elif self.input_dim >= 3:
                # 3D rotation around z-axis, identity for higher dimensions
                rotation = np.eye(self.input_dim)
                if self.input_dim >= 2:
                    rotation[0, 0] = math.cos(angle)
                    rotation[0, 1] = -math.sin(angle)
                    rotation[1, 0] = math.sin(angle)
                    rotation[1, 1] = math.cos(angle)
            else:
                # 1D case - just scaling
                rotation = np.array([[(-1) ** k]])
                
            group_elements.append(rotation)
            
        return group_elements
    
    def _generate_dihedral_group(self, n: int) -> List[np.ndarray]:
        """Generate dihedral group Dn (rotations + reflections)."""
        group_elements = []
        
        # Add rotations (same as cyclic group)
        rotations = self._generate_cyclic_group(n)
        group_elements.extend(rotations)
        
        # Add reflections
        for k in range(n):
            angle = 2 * math.pi * k / n
            
            if self.input_dim == 2:
                # Reflection across line through origin at angle/2
                reflection_angle = angle / 2
                cos_2theta = math.cos(2 * reflection_angle)
                sin_2theta = math.sin(2 * reflection_angle)
                
                reflection = np.array([
                    [cos_2theta, sin_2theta],
                    [sin_2theta, -cos_2theta]
                ])
            else:
                # For higher dimensions, reflect across first coordinate
                reflection = np.eye(self.input_dim)
                reflection[0, 0] = -1
                
            group_elements.append(reflection)
            
        return group_elements
    
    def _generate_symmetric_group_3(self) -> List[np.ndarray]:
        """Generate symmetric group S3 (permutations of 3 elements)."""
        if self.input_dim < 3:
            # For lower dimensions, use reduced representation
            return self._generate_cyclic_group(3)
        
        # All permutations of 3 elements as permutation matrices
        permutations = [
            [0, 1, 2],  # identity
            [1, 2, 0],  # (0 1 2)
            [2, 0, 1],  # (0 2 1)
            [1, 0, 2],  # (0 1)
            [0, 2, 1],  # (0 2)
            [2, 1, 0],  # (1 2)
        ]
        
        group_elements = []
        for perm in permutations:
            matrix = np.eye(self.input_dim)
            for i in range(3):
                if i < self.input_dim and perm[i] < self.input_dim:
                    matrix[i, i] = 0
                    matrix[i, perm[i]] = 1
            group_elements.append(matrix)
            
        return group_elements
    
    def _generate_reflection_group(self) -> List[np.ndarray]:
        """Generate group of reflections across coordinate axes."""
        group_elements = []
        
        # Identity
        group_elements.append(np.eye(self.input_dim))
        
        # Reflections across each coordinate axis
        for i in range(self.input_dim):
            reflection = np.eye(self.input_dim)
            reflection[i, i] = -1
            group_elements.append(reflection)
            
        # Reflections across diagonal planes (for 2D and 3D)
        if self.input_dim == 2:
            # Reflection across y = x
            diag_reflection = np.array([[0, 1], [1, 0]])
            group_elements.append(diag_reflection)
            
            # Reflection across y = -x
            anti_diag_reflection = np.array([[0, -1], [-1, 0]])
            group_elements.append(anti_diag_reflection)
            
        return group_elements
    
    def apply_group_actions(self, x: np.ndarray, group_name: str) -> np.ndarray:
        """Apply all group actions to input data."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        group_elements = self.groups[group_name]
        results = []
        
        for element in group_elements:
            if x.shape[1] == element.shape[0]:
                transformed = x @ element.T
            else:
                # Handle dimension mismatch by padding or truncating
                min_dim = min(x.shape[1], element.shape[0])
                transformed = x[:, :min_dim] @ element[:min_dim, :min_dim].T
                
            # Compute invariant features
            norm = np.linalg.norm(transformed, axis=1, keepdims=True)
            mean = np.mean(transformed, axis=1, keepdims=True)
            std = np.std(transformed, axis=1, keepdims=True) + 1e-8
            
            # Combine features
            features = np.concatenate([norm, mean, std], axis=1)
            results.append(features)
            
        return np.concatenate(results, axis=1)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through group theory network."""
        all_features = []
        
        # Apply each group type
        for group_name in self.group_types:
            group_features = self.apply_group_actions(x, group_name)
            all_features.append(group_features)
            
        # Concatenate all features
        combined_features = np.concatenate(all_features, axis=1)
        
        # Linear combination to get desired output dimension
        feature_dim = combined_features.shape[1]
        if feature_dim >= self.output_dim:
            # Take first output_dim features
            output = combined_features[:, :self.output_dim]
        else:
            # Repeat features to reach output_dim
            repeats = (self.output_dim + feature_dim - 1) // feature_dim
            repeated = np.tile(combined_features, (1, repeats))
            output = repeated[:, :self.output_dim]
            
        return output
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Prediction method."""
        return self.forward(x)


def test_rotation_invariance():
    """Test rotation invariance of the group theory network."""
    print("=== Group Theory Network: Rotation Invariance Test ===\n")
    
    # Create network with cyclic group
    network = GroupTheoryNetwork(
        input_dim=2,
        group_types=["cyclic_8"],
        output_dim=4
    )
    
    # Create test patterns
    original_pattern = np.array([[1, 0], [0, 1], [1, 1]])
    
    # Manual rotations for comparison
    rotation_angles = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
    
    print("Testing rotation invariance:")
    outputs = []
    
    for angle in rotation_angles:
        # Manually rotate the pattern
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_pattern = original_pattern @ rotation_matrix.T
        
        # Get network output
        output = network.predict(rotated_pattern)
        outputs.append(output)
        
        print(f"  Rotation {angle:.2f} rad: mean output = {np.mean(output):.4f}")
    
    # Check invariance (outputs should be similar)
    output_array = np.array(outputs)
    variance_across_rotations = np.var(output_array, axis=0)
    mean_variance = np.mean(variance_across_rotations)
    
    print(f"\nMean variance across rotations: {mean_variance:.6f}")
    print("(Lower values indicate better rotation invariance)\n")
    
    return outputs


def test_symmetry_detection():
    """Test the network's ability to detect different symmetries."""
    print("=== Group Theory Network: Symmetry Detection ===\n")
    
    # Create network with multiple group types
    network = GroupTheoryNetwork(
        input_dim=2,
        group_types=["cyclic_4", "dihedral_4", "reflection"],
        output_dim=6
    )
    
    # Define test patterns with different symmetries
    patterns = {
        "Square": np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]),
        "Triangle": np.array([[1, 0], [-0.5, np.sqrt(3)/2], [-0.5, -np.sqrt(3)/2]]),
        "Line": np.array([[1, 0], [0.5, 0], [0, 0], [-0.5, 0], [-1, 0]]),
        "Circle": np.array([[np.cos(θ), np.sin(θ)] for θ in np.linspace(0, 2*np.pi, 8)]),
        "Asymmetric": np.array([[1, 0], [0, 1], [0.3, 0.7], [0.8, 0.2]])
    }
    
    print("Symmetry detection results:")
    pattern_outputs = {}
    
    for pattern_name, pattern_points in patterns.items():
        output = network.predict(pattern_points)
        pattern_outputs[pattern_name] = output
        
        print(f"\n{pattern_name}:")
        print(f"  Mean output: {np.mean(output, axis=0)}")
        print(f"  Output std: {np.std(output, axis=0)}")
        
    return pattern_outputs


def test_group_composition():
    """Test composition of group operations."""
    print("=== Group Theory Network: Group Composition ===\n")
    
    network = GroupTheoryNetwork(
        input_dim=3,
        group_types=["symmetric_3"],
        output_dim=3
    )
    
    # Test group properties
    test_input = np.array([[1, 2, 3]])
    
    print("Testing group composition properties:")
    
    # Get all group elements
    group_elements = network.groups["symmetric_3"]
    
    # Test identity element (should be first)
    identity_result = test_input @ group_elements[0].T
    print(f"Identity transformation: {test_input[0]} → {identity_result[0]}")
    
    # Test composition of operations
    for i, g1 in enumerate(group_elements[:3]):
        for j, g2 in enumerate(group_elements[:3]):
            # Apply g1 then g2
            intermediate = test_input @ g1.T
            final = intermediate @ g2.T
            
            # Apply composition g2∘g1
            composition = g2 @ g1
            direct = test_input @ composition.T
            
            # Check if they're the same (within numerical precision)
            difference = np.linalg.norm(final - direct)
            print(f"  g{j}∘g{i}: composition error = {difference:.8f}")


def test_invariant_features():
    """Test extraction of invariant features."""
    print("=== Group Theory Network: Invariant Feature Extraction ===\n")
    
    # Network for extracting rotation-invariant features
    network = GroupTheoryNetwork(
        input_dim=2,
        group_types=["cyclic_16"],  # Fine rotation sampling
        output_dim=8
    )
    
    # Test patterns with known geometric properties
    test_cases = [
        ("Unit Circle", np.array([[np.cos(θ), np.sin(θ)] for θ in np.linspace(0, 2*np.pi, 10)])),
        ("Ellipse", np.array([[2*np.cos(θ), np.sin(θ)] for θ in np.linspace(0, 2*np.pi, 10)])),
        ("Square", np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])),
        ("Random", np.random.randn(8, 2))
    ]
    
    print("Invariant feature analysis:")
    
    for case_name, points in test_cases:
        # Original features
        original_features = network.predict(points)
        
        # Rotated version
        rotation_45 = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                               [np.sin(np.pi/4), np.cos(np.pi/4)]])
        rotated_points = points @ rotation_45.T
        rotated_features = network.predict(rotated_points)
        
        # Measure feature consistency
        feature_difference = np.linalg.norm(original_features - rotated_features)
        relative_difference = feature_difference / (np.linalg.norm(original_features) + 1e-8)
        
        print(f"\n{case_name}:")
        print(f"  Feature difference: {feature_difference:.6f}")
        print(f"  Relative difference: {relative_difference:.6f}")
        print(f"  Original features mean: {np.mean(original_features):.4f}")
        print(f"  Rotated features mean: {np.mean(rotated_features):.4f}")


if __name__ == "__main__":
    print("Group Theory Algebraic Neural Network Demo\n")
    print("="*60)
    
    # Run tests
    test_rotation_invariance()
    test_symmetry_detection()
    test_group_composition()
    test_invariant_features()
    
    print("\n" + "="*60)
    print("Group theory demo completed successfully!")