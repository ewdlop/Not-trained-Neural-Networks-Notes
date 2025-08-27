"""
Lazy Neural Network Implementation

This module implements lazy neural networks that use deferred computation.
Calculations are only performed when results are actually needed, providing
memory efficiency and computational optimization.

Author: Lazy Neural Network Research
"""

import numpy as np
from typing import List, Callable, Union, Optional, Any, Dict
import math
from functools import wraps
import weakref


class LazyValue:
    """
    Represents a lazy-evaluated value that is computed only when needed.
    """
    
    def __init__(self, computation_func: Callable, *args, **kwargs):
        """
        Initialize a lazy value.
        
        Args:
            computation_func: Function to compute the value
            *args, **kwargs: Arguments for the computation function
        """
        self._computation_func = computation_func
        self._args = args
        self._kwargs = kwargs
        self._computed = False
        self._value = None
        self._computation_count = 0
    
    def compute(self) -> Any:
        """Compute and return the value."""
        if not self._computed:
            self._value = self._computation_func(*self._args, **self._kwargs)
            self._computed = True
            self._computation_count += 1
        return self._value
    
    @property
    def value(self) -> Any:
        """Get the computed value (alias for compute)."""
        return self.compute()
    
    @property
    def is_computed(self) -> bool:
        """Check if the value has been computed."""
        return self._computed
    
    @property
    def computation_count(self) -> int:
        """Get the number of times this value was computed."""
        return self._computation_count
    
    def reset(self):
        """Reset the lazy value to uncomputed state."""
        self._computed = False
        self._value = None


class LazyCache:
    """
    Cache for lazy computations with automatic cleanup.
    """
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: Dict[str, LazyValue] = {}
        self._access_order: List[str] = []
    
    def get(self, key: str) -> Optional[LazyValue]:
        """Get a cached lazy value."""
        if key in self._cache:
            # Move to end of access order (LRU)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def put(self, key: str, lazy_value: LazyValue):
        """Store a lazy value in cache."""
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self.max_size:
            # Remove least recently used
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
        
        self._cache[key] = lazy_value
        self._access_order.append(key)
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


def lazy_computation(func):
    """
    Decorator to make a function lazy-evaluated.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return LazyValue(func, *args, **kwargs)
    return wrapper


class LazyLayer:
    """
    Base class for lazy neural network layers that implement deferred computation.
    """
    
    def __init__(self, input_size: int, output_size: int, layer_type: str = "lazy"):
        self.input_size = input_size
        self.output_size = output_size
        self.layer_type = layer_type
        self._cache = LazyCache()
        self._computation_stats = {
            'forward_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def _generate_cache_key(self, x: np.ndarray) -> str:
        """Generate a cache key for input data."""
        # Use hash of input array as cache key
        return str(hash(x.data.tobytes() if hasattr(x, 'data') else x.tobytes()))
    
    def forward_lazy(self, x: np.ndarray) -> LazyValue:
        """
        Lazy forward pass that returns a LazyValue.
        
        Args:
            x: Input data
            
        Returns:
            LazyValue that will compute the forward pass when evaluated
        """
        self._computation_stats['forward_calls'] += 1
        
        # Check cache first
        cache_key = self._generate_cache_key(x)
        cached_result = self._cache.get(cache_key)
        
        if cached_result is not None:
            self._computation_stats['cache_hits'] += 1
            return cached_result
        
        self._computation_stats['cache_misses'] += 1
        
        # Create lazy computation
        lazy_result = LazyValue(self._compute_forward, x)
        self._cache.put(cache_key, lazy_result)
        
        return lazy_result
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Standard forward pass that immediately computes the result.
        """
        return self.forward_lazy(x).compute()
    
    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Internal method to actually compute the forward pass.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _compute_forward")
    
    def reset_cache(self):
        """Reset the layer's cache."""
        self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get computation statistics."""
        return {
            **self._computation_stats,
            'cache_size': self._cache.size(),
            'cache_hit_rate': (
                self._computation_stats['cache_hits'] / 
                max(1, self._computation_stats['cache_hits'] + self._computation_stats['cache_misses'])
            )
        }


class LazyPolynomialLayer(LazyLayer):
    """
    Lazy polynomial layer that computes polynomial transformations on-demand.
    """
    
    def __init__(self, input_size: int, output_size: int, degree: int = 2):
        super().__init__(input_size, output_size, "lazy_polynomial")
        self.degree = degree
        self._coefficients = None
    
    @property
    def coefficients(self) -> np.ndarray:
        """Get polynomial coefficients (computed lazily)."""
        if self._coefficients is None:
            self._coefficients = self._generate_coefficients()
        return self._coefficients
    
    def _generate_coefficients(self) -> np.ndarray:
        """Generate polynomial coefficients using algebraic sequences."""
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        coeffs = []
        
        for i in range(self.output_size):
            for j in range(self.input_size):
                coeff = (phi ** (i + 1)) / (j + 1) if j < self.input_size else 1.0
                coeffs.append(coeff)
                
        return np.array(coeffs).reshape(self.output_size, self.input_size)
    
    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        """Compute polynomial transformation."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        result = np.zeros((x.shape[0], self.output_size))
        
        for i in range(self.output_size):
            poly_result = np.zeros(x.shape[0])
            for degree in range(1, self.degree + 1):
                poly_term = np.power(x, degree) @ self.coefficients[i]
                poly_result += poly_term / math.factorial(degree)
            result[:, i] = poly_result
            
        return result


class LazyGroupTheoryLayer(LazyLayer):
    """
    Lazy group theory layer that applies group operations on-demand.
    """
    
    def __init__(self, input_size: int, output_size: int, group_order: int = 8):
        super().__init__(input_size, output_size, "lazy_group_theory")
        self.group_order = group_order
        self._group_elements = None
    
    @property
    def group_elements(self) -> List[np.ndarray]:
        """Get group elements (computed lazily)."""
        if self._group_elements is None:
            self._group_elements = self._generate_group_elements()
        return self._group_elements
    
    def _generate_group_elements(self) -> List[np.ndarray]:
        """Generate cyclic group elements as rotation matrices."""
        elements = []
        for k in range(self.group_order):
            angle = 2 * math.pi * k / self.group_order
            if self.input_size == 2:
                rotation = np.array([
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)]
                ])
                elements.append(rotation)
            else:
                rotation = np.eye(self.input_size)
                if self.input_size >= 2:
                    rotation[0, 0] = math.cos(angle)
                    rotation[0, 1] = -math.sin(angle)
                    rotation[1, 0] = math.sin(angle)
                    rotation[1, 1] = math.cos(angle)
                elements.append(rotation)
        return elements
    
    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        """Apply group action to input."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        results = []
        for i, group_element in enumerate(self.group_elements[:self.output_size]):
            if x.shape[1] == group_element.shape[0]:
                transformed = x @ group_element.T
                result = np.linalg.norm(transformed, axis=1)
            else:
                result = np.sum(x * (i + 1), axis=1)
            results.append(result)
            
        return np.column_stack(results)


class LazyGeometricAlgebraLayer(LazyLayer):
    """
    Lazy geometric algebra layer that computes geometric products on-demand.
    """
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size, "lazy_geometric_algebra")
        self._basis_vectors = None
    
    @property
    def basis_vectors(self) -> List[np.ndarray]:
        """Get basis vectors (computed lazily)."""
        if self._basis_vectors is None:
            self._basis_vectors = self._generate_basis_vectors()
        return self._basis_vectors
    
    def _generate_basis_vectors(self) -> List[np.ndarray]:
        """Generate basis vectors for geometric algebra."""
        basis = []
        # Scalar basis
        basis.append(np.ones(self.input_size))
        
        # Vector basis
        for i in range(self.input_size):
            e_i = np.zeros(self.input_size)
            e_i[i] = 1.0
            basis.append(e_i)
            
        # Bivector basis (for pairs)
        for i in range(self.input_size):
            for j in range(i + 1, self.input_size):
                e_ij = np.zeros(self.input_size)
                e_ij[i] = 1.0
                e_ij[j] = 1.0
                basis.append(e_ij)
                
        return basis[:self.output_size]
    
    def _geometric_product(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute geometric product of two vectors."""
        # Ensure vectors have same length for dot product
        min_len = min(len(a), len(b))
        a_truncated = a[:min_len]
        b_truncated = b[:min_len]
        
        dot_prod = np.dot(a_truncated, b_truncated)
        outer_norm = np.linalg.norm(np.outer(a_truncated, b_truncated))
        return dot_prod + outer_norm
    
    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        """Apply geometric algebra transformations."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        results = []
        for basis_vector in self.basis_vectors:
            result = []
            for sample in x:
                gp = self._geometric_product(sample, basis_vector)
                result.append(gp)
            results.append(np.array(result))
            
        return np.column_stack(results)


class LazyNeuralNetwork:
    """
    Lazy neural network that orchestrates deferred computation across layers.
    """
    
    def __init__(self):
        self.layers: List[LazyLayer] = []
        self._computation_stats = {
            'forward_calls': 0,
            'lazy_evaluations': 0,
            'immediate_evaluations': 0
        }
    
    def add_layer(self, layer: LazyLayer):
        """Add a lazy layer to the network."""
        self.layers.append(layer)
    
    def forward_lazy(self, x: np.ndarray) -> LazyValue:
        """
        Lazy forward pass that returns a LazyValue for the final result.
        """
        self._computation_stats['forward_calls'] += 1
        self._computation_stats['lazy_evaluations'] += 1
        
        return LazyValue(self._compute_full_forward, x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Standard forward pass that immediately computes the result."""
        self._computation_stats['forward_calls'] += 1
        self._computation_stats['immediate_evaluations'] += 1
        
        return self._compute_full_forward(x)
    
    def _compute_full_forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the full forward pass through all layers."""
        current_input = x
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Alias for forward pass."""
        return self.forward(x)
    
    def predict_lazy(self, x: np.ndarray) -> LazyValue:
        """Alias for lazy forward pass."""
        return self.forward_lazy(x)
    
    def partial_predict(self, x: np.ndarray, layer_indices: List[int]) -> List[np.ndarray]:
        """
        Compute predictions only for specified layers.
        
        Args:
            x: Input data
            layer_indices: List of layer indices to compute
            
        Returns:
            List of outputs for the specified layers
        """
        if not layer_indices:
            return []
        
        # Sort indices to compute in order
        sorted_indices = sorted(set(layer_indices))
        max_index = max(sorted_indices)
        
        if max_index >= len(self.layers):
            raise ValueError(f"Layer index {max_index} out of range")
        
        # Compute forward pass up to the maximum required layer
        current_input = x
        layer_outputs = {}
        
        for i in range(max_index + 1):
            current_input = self.layers[i].forward(current_input)
            if i in sorted_indices:
                layer_outputs[i] = current_input.copy()
        
        # Return outputs in the requested order
        return [layer_outputs[i] for i in layer_indices]
    
    def reset_caches(self):
        """Reset all layer caches."""
        for layer in self.layers:
            layer.reset_cache()
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get comprehensive network statistics."""
        layer_stats = []
        for i, layer in enumerate(self.layers):
            layer_stats.append({
                'layer_index': i,
                'layer_type': layer.layer_type,
                **layer.get_stats()
            })
        
        total_cache_hits = sum(ls['cache_hits'] for ls in layer_stats)
        total_cache_misses = sum(ls['cache_misses'] for ls in layer_stats)
        total_cache_size = sum(ls['cache_size'] for ls in layer_stats)
        
        return {
            'network_stats': self._computation_stats,
            'layer_stats': layer_stats,
            'total_cache_hits': total_cache_hits,
            'total_cache_misses': total_cache_misses,
            'total_cache_size': total_cache_size,
            'overall_cache_hit_rate': (
                total_cache_hits / max(1, total_cache_hits + total_cache_misses)
            )
        }


def create_sample_lazy_network() -> LazyNeuralNetwork:
    """Create a sample lazy neural network for demonstration."""
    network = LazyNeuralNetwork()
    
    # Add different types of lazy algebraic layers
    network.add_layer(LazyPolynomialLayer(4, 6, degree=2))
    network.add_layer(LazyGroupTheoryLayer(6, 4, group_order=8))
    network.add_layer(LazyGeometricAlgebraLayer(4, 2))
    
    return network


def demo_lazy_neural_network():
    """Demonstrate the lazy neural network with sample data."""
    print("=== Lazy Neural Network Demo ===\n")
    
    # Create sample network
    network = create_sample_lazy_network()
    
    # Generate sample input data
    np.random.seed(42)
    sample_input = np.random.randn(5, 4)
    
    print("Input data shape:", sample_input.shape)
    print("Input data:\n", sample_input)
    
    # Demonstrate lazy evaluation
    print("\n=== Lazy Evaluation Demo ===")
    
    # Create lazy prediction
    lazy_prediction = network.predict_lazy(sample_input)
    print(f"Lazy prediction created: {lazy_prediction}")
    print(f"Is computed: {lazy_prediction.is_computed}")
    
    # Compute the result
    output = lazy_prediction.compute()
    print(f"After computation, is computed: {lazy_prediction.is_computed}")
    print(f"Output shape: {output.shape}")
    print("Output data:\n", output)
    
    # Demonstrate immediate evaluation
    print("\n=== Immediate Evaluation Demo ===")
    immediate_output = network.predict(sample_input)
    print(f"Immediate output shape: {immediate_output.shape}")
    
    # Verify they're the same
    difference = np.linalg.norm(output - immediate_output)
    print(f"Difference between lazy and immediate: {difference:.10f}")
    
    # Demonstrate partial prediction
    print("\n=== Partial Prediction Demo ===")
    partial_outputs = network.partial_predict(sample_input[:2], [0, 2])
    print(f"Partial predictions for layers 0 and 2:")
    for i, partial_out in enumerate(partial_outputs):
        print(f"  Layer output {i}: shape {partial_out.shape}")
    
    # Show statistics
    print("\n=== Network Statistics ===")
    stats = network.get_network_stats()
    print(f"Network stats: {stats['network_stats']}")
    print(f"Overall cache hit rate: {stats['overall_cache_hit_rate']:.2%}")
    print(f"Total cache size: {stats['total_cache_size']}")


if __name__ == "__main__":
    demo_lazy_neural_network()