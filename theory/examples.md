# Worked Examples of Algebraic Neural Networks

## Example 1: Polynomial Network for Function Approximation

### Problem
Approximate the function f(x, y) = x² + 2xy + y² using an algebraic neural network.

### Solution

```python
import numpy as np
from algebraic_neural_network import PolynomialLayer

# Create a polynomial layer
poly_layer = PolynomialLayer(input_size=2, output_size=1, degree=2)

# Test points
test_points = np.array([
    [1, 1],    # f(1,1) = 1 + 2 + 1 = 4
    [2, 3],    # f(2,3) = 4 + 12 + 9 = 25
    [0, 1],    # f(0,1) = 0 + 0 + 1 = 1
    [-1, 2]    # f(-1,2) = 1 - 4 + 4 = 1
])

# Apply the polynomial transformation
output = poly_layer.forward(test_points)
print("Polynomial approximation results:", output.flatten())
```

### Mathematical Analysis
The polynomial layer generates coefficients using the golden ratio φ = (1 + √5)/2:
- For output neuron 1: coefficient matrix uses φ¹ and φ²
- The transformation applies: y = Σᵢ Σⱼ (aᵢⱼ/j!) xʲ

## Example 2: Group Theory Network for Rotation Invariance

### Problem
Create a network that recognizes patterns invariant under rotations.

### Solution

```python
import numpy as np
from algebraic_neural_network import GroupTheoryLayer

# Create a group theory layer using 8-fold rotational symmetry
group_layer = GroupTheoryLayer(input_size=2, output_size=4, group_order=8)

# Test with a simple pattern (vector pointing in different directions)
patterns = np.array([
    [1, 0],      # Point along x-axis
    [0, 1],      # Point along y-axis
    [1/√2, 1/√2], # Point along 45° diagonal
    [-1, 0]      # Point along negative x-axis
])

# Apply group transformations
transformed = group_layer.forward(patterns)
print("Group theory transformation results:")
print(transformed)
```

### Mathematical Analysis
The group theory layer applies rotations from the cyclic group C₈:
- Rotation angles: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
- Each rotation is represented by a 2×2 rotation matrix
- The norm of the transformed vector provides rotation-invariant features

## Example 3: Geometric Algebra Network for 3D Processing

### Problem
Process 3D geometric data using geometric algebra operations.

### Solution

```python
import numpy as np
from algebraic_neural_network import GeometricAlgebraLayer

# Create geometric algebra layer
geo_layer = GeometricAlgebraLayer(input_size=3, output_size=4)

# 3D vectors representing different geometric entities
vectors = np.array([
    [1, 0, 0],    # Unit vector along x
    [0, 1, 0],    # Unit vector along y  
    [0, 0, 1],    # Unit vector along z
    [1, 1, 1]     # Diagonal vector
])

# Apply geometric algebra transformations
geo_output = geo_layer.forward(vectors)
print("Geometric algebra results:")
print(geo_output)
```

### Mathematical Analysis
The geometric algebra layer computes:
- Scalar products: a·b
- Vector products: a∧b (bivectors)
- Trivector products: a∧b∧c
- Mixed products combining all grades

## Example 4: Complete Network for Pattern Classification

### Problem
Build a complete algebraic neural network for classifying 2D patterns.

### Solution

```python
from algebraic_neural_network import AlgebraicNeuralNetwork, PolynomialLayer, GroupTheoryLayer

# Create network architecture
network = AlgebraicNeuralNetwork()
network.add_layer(PolynomialLayer(2, 4, degree=2))      # Feature extraction
network.add_layer(GroupTheoryLayer(4, 3, group_order=6)) # Symmetry processing
network.add_layer(PolynomialLayer(3, 1, degree=1))      # Final classification

# Test patterns
circle_points = np.array([
    [np.cos(θ), np.sin(θ)] for θ in np.linspace(0, 2*np.pi, 8)
])

square_points = np.array([
    [1, 1], [1, -1], [-1, -1], [-1, 1],
    [1, 0], [0, 1], [-1, 0], [0, -1]
])

# Classify patterns
circle_scores = network.predict(circle_points)
square_scores = network.predict(square_points)

print("Circle pattern scores:", np.mean(circle_scores))
print("Square pattern scores:", np.mean(square_scores))
```

### Analysis
This network demonstrates:
1. **Feature Extraction**: Polynomial layer extracts nonlinear features
2. **Symmetry Processing**: Group theory layer handles rotational symmetries
3. **Classification**: Final layer provides decision boundary

## Example 5: Time Series Processing with Algebraic Networks

### Problem
Process time series data using algebraic transformations.

### Solution

```python
import numpy as np

def create_time_series_network():
    network = AlgebraicNeuralNetwork()
    # Window-based polynomial features
    network.add_layer(PolynomialLayer(5, 6, degree=2))  # 5-point window
    # Temporal symmetries
    network.add_layer(GroupTheoryLayer(6, 4, group_order=4))
    # Final prediction
    network.add_layer(PolynomialLayer(4, 1, degree=1))
    return network

# Generate sample time series
t = np.linspace(0, 4*np.pi, 100)
signal = np.sin(t) + 0.3*np.sin(3*t) + 0.1*np.random.randn(100)

# Create windows of 5 consecutive points
windows = np.array([signal[i:i+5] for i in range(len(signal)-4)])

# Process with algebraic network
ts_network = create_time_series_network()
predictions = ts_network.predict(windows)

print(f"Processed {len(windows)} time windows")
print(f"Prediction range: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")
```

### Analysis
This demonstrates algebraic networks for temporal data:
- **Windowing**: Convert time series to fixed-size vectors
- **Polynomial Features**: Capture local nonlinear patterns
- **Temporal Symmetries**: Handle time-shift invariances

## Example 6: Supersymmetric Network for Physics-Inspired Processing

### Problem
Process data using supersymmetric algebraic structures that separate bosonic and fermionic components with anticommuting Grassmann algebra.

### Solution

```python
import numpy as np
from algebraic_neural_network import SupersymmetryLayer, AlgebraicNeuralNetwork

# Create supersymmetric layer
susy_layer = SupersymmetryLayer(input_size=4, output_size=3, n_grassmann=2)

# Input with bosonic and fermionic components
bosonic_data = np.array([1.0, 0.5, -0.3, 0.8])  # Commuting components
fermionic_data = np.array([0.1, -0.2, 0.4, -0.1]) # Anticommuting components

# Combine into superspace coordinates
superspace_input = np.concatenate([bosonic_data[:2], fermionic_data[:2]])

# Process through supersymmetric transformations
output = susy_layer.forward(superspace_input.reshape(1, -1))

# Test Grassmann algebra properties
theta1 = np.array([1, 0, 0, 0])
theta2 = np.array([0, 1, 0, 0])

# Anticommutation: θ₁∧θ₂ = -θ₂∧θ₁
ab = susy_layer.grassmann_product(theta1, theta2)
ba = susy_layer.grassmann_product(theta2, theta1)
print(f"Anticommutation check: {ab + ba} (should be ≈0)")

# Nilpotent property: θ² = 0
aa = susy_layer.grassmann_product(theta1, theta1)
print(f"Nilpotent property: {aa} (should be 0)")
```

### Mathematical Analysis
The supersymmetric layer implements:

**Grassmann Algebra**: Variables satisfy θᵢθⱼ = -θⱼθᵢ and θᵢ² = 0

**Supersymmetric Transformations**: 
- δφ = εψ (bosonic field gains fermionic contribution)
- δψ = ε∂φ (fermionic field gains bosonic derivative)

**Anticommuting Operations**: Natural encoding of fermionic statistics from particle physics

**Bosonic/Fermionic Duality**: Systematic treatment of two fundamental particle types

This demonstrates how theoretical physics concepts can be incorporated into neural network architectures to create novel computational frameworks with built-in physical symmetries.

## Performance Characteristics

### Computational Complexity
- **Polynomial Layers**: O(nd) where n is input size, d is degree
- **Group Theory Layers**: O(ng) where g is group order
- **Supersymmetry Layers**: O(n²g) where g is number of Grassmann variables
- **Geometric Algebra Layers**: O(n²) for geometric products

### Memory Requirements
- **Fixed Coefficients**: No weight storage needed
- **Intermediate Results**: Only temporary computation storage
- **Total Memory**: O(n) where n is largest layer size

### Accuracy Analysis
Algebraic networks provide:
- **Consistency**: Same input always produces same output
- **Stability**: Small input changes → small output changes
- **Interpretability**: Mathematical meaning for each operation

## Practical Considerations

### When to Use Algebraic Networks
- **Known Mathematical Structure**: Problem has clear algebraic properties
- **No Training Data**: When gradient-based training isn't feasible
- **Interpretability Required**: Need mathematical understanding of operations
- **Real-time Processing**: Fast, deterministic computation needed

### Limitations
- **Limited Expressivity**: May not capture all possible patterns
- **Parameter Selection**: Choosing group orders, polynomial degrees
- **Scaling**: Performance with very high-dimensional data

### Extensions
- **Adaptive Coefficients**: Use algebraic sequences that adapt to data
- **Hybrid Networks**: Combine with traditional neural networks
- **Custom Algebras**: Develop problem-specific algebraic structures