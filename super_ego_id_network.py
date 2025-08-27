"""
Super-Ego-Id Neural Network Implementation in PyTorch

This module implements a neural network architecture inspired by Freudian psychoanalytic theory,
consisting of three components representing the Id, Ego, and Super-ego.

The architecture demonstrates how psychological concepts can be translated into neural network
design, with each component having distinct roles in the decision-making process.

Components:
- Id Network: Represents immediate, instinctual responses (pleasure principle)
- Ego Network: Mediates between id and super-ego, handles realistic constraints
- Super-ego Network: Applies moral constraints and learned rules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class IdNetwork(nn.Module):
    """
    Id Network: Represents the primitive, instinctual component.
    
    This component focuses on immediate gratification and basic responses,
    similar to the id in Freudian theory which operates on the pleasure principle.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(IdNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with high activation to represent impulsive responses."""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # Use higher activation (less constrained) to represent impulsive nature
        x = torch.tanh(self.fc3(x)) * 2.0  
        return x


class EgoNetwork(nn.Module):
    """
    Ego Network: Represents the realistic, mediating component.
    
    This component balances the demands of the id and super-ego,
    operating on the reality principle to find practical solutions.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(EgoNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with balanced activation representing realistic responses."""
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        # Balanced activation representing realistic constraints
        x = torch.tanh(self.fc3(x))
        return x


class SuperEgoNetwork(nn.Module):
    """
    Super-ego Network: Represents the moral, constraining component.
    
    This component applies ethical constraints and learned social rules,
    similar to the super-ego which represents internalized moral standards.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(SuperEgoNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.constraint_layer = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with constrained activation representing moral guidelines."""
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.1)
        x = self.dropout(x)
        x = torch.tanh(self.fc3(x))
        # Apply additional constraints representing moral limitations
        constraints = torch.sigmoid(self.constraint_layer(x))
        x = x * constraints * 0.5  # More constrained output
        return x


class SuperEgoIdNetwork(nn.Module):
    """
    Complete Super-Ego-Id Neural Network.
    
    This network combines the three psychoanalytic components (Id, Ego, Super-ego)
    into a unified architecture that demonstrates how psychological concepts can
    influence neural network design and decision-making processes.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 10):
        super(SuperEgoIdNetwork, self).__init__()
        
        # Three psychological components
        self.id_network = IdNetwork(input_dim, hidden_dim, output_dim)
        self.ego_network = EgoNetwork(input_dim, hidden_dim, output_dim)
        self.superego_network = SuperEgoNetwork(input_dim, hidden_dim, output_dim)
        
        # Integration layer to combine all three components
        self.integration_layer = nn.Linear(output_dim * 3, output_dim)
        self.final_layer = nn.Linear(output_dim, output_dim)
        
        # Attention mechanism to weight the importance of each component
        # Ensure embed_dim is divisible by num_heads
        attention_heads = min(4, output_dim)  # Use fewer heads if output_dim is small
        while output_dim % attention_heads != 0 and attention_heads > 1:
            attention_heads -= 1
        self.attention = nn.MultiheadAttention(output_dim, num_heads=attention_heads, batch_first=True)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through all three components with integration.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (final_output, component_outputs)
        """
        # Get outputs from each psychological component
        id_output = self.id_network(x)
        ego_output = self.ego_network(x)
        superego_output = self.superego_network(x)
        
        # Store individual outputs for analysis
        component_outputs = {
            'id': id_output,
            'ego': ego_output,
            'superego': superego_output
        }
        
        # Concatenate all outputs
        combined = torch.cat([id_output, ego_output, superego_output], dim=-1)
        integrated = F.relu(self.integration_layer(combined))
        
        # Apply attention mechanism to balance components
        # Reshape for attention: (batch_size, seq_len=3, feature_dim)
        batch_size = x.size(0)
        attention_input = torch.stack([id_output, ego_output, superego_output], dim=1)
        
        attended_output, attention_weights = self.attention(
            attention_input, attention_input, attention_input
        )
        
        # Average the attended outputs
        attended_combined = torch.mean(attended_output, dim=1)
        
        # Combine integrated and attended outputs
        final_output = self.final_layer(integrated + attended_combined)
        
        # Store attention weights for interpretability
        component_outputs['attention_weights'] = attention_weights
        
        return final_output, component_outputs
    
    def get_component_analysis(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze the contribution of each psychological component.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary containing analysis of each component's contribution
        """
        with torch.no_grad():
            final_output, component_outputs = self.forward(x)
            
            analysis = {
                'id_magnitude': torch.mean(torch.abs(component_outputs['id'])).item(),
                'ego_magnitude': torch.mean(torch.abs(component_outputs['ego'])).item(),
                'superego_magnitude': torch.mean(torch.abs(component_outputs['superego'])).item(),
                'attention_weights': component_outputs['attention_weights'],
                'final_output': final_output,
                'component_outputs': component_outputs
            }
            
            # Calculate dominance scores
            total_magnitude = (analysis['id_magnitude'] + 
                             analysis['ego_magnitude'] + 
                             analysis['superego_magnitude'])
            
            if total_magnitude > 0:
                analysis['id_dominance'] = analysis['id_magnitude'] / total_magnitude
                analysis['ego_dominance'] = analysis['ego_magnitude'] / total_magnitude
                analysis['superego_dominance'] = analysis['superego_magnitude'] / total_magnitude
            else:
                analysis['id_dominance'] = analysis['ego_dominance'] = analysis['superego_dominance'] = 0.0
            
            return analysis


def create_sample_data(batch_size: int = 32, input_dim: int = 20) -> torch.Tensor:
    """Create sample data for testing the network."""
    return torch.randn(batch_size, input_dim)


def demonstrate_network():
    """
    Demonstrate the Super-Ego-Id Network with sample data.
    """
    print("Super-Ego-Id Neural Network Demonstration")
    print("=" * 50)
    
    # Create network
    input_dim = 20
    hidden_dim = 64
    output_dim = 10
    
    network = SuperEgoIdNetwork(input_dim, hidden_dim, output_dim)
    
    # Create sample data
    sample_input = create_sample_data(batch_size=5, input_dim=input_dim)
    
    # Forward pass
    output, components = network(sample_input)
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print()
    
    # Analyze components
    analysis = network.get_component_analysis(sample_input)
    
    print("Component Analysis:")
    print(f"Id dominance: {analysis['id_dominance']:.3f}")
    print(f"Ego dominance: {analysis['ego_dominance']:.3f}")
    print(f"Super-ego dominance: {analysis['superego_dominance']:.3f}")
    print()
    
    print("Component Magnitudes:")
    print(f"Id magnitude: {analysis['id_magnitude']:.3f}")
    print(f"Ego magnitude: {analysis['ego_magnitude']:.3f}")
    print(f"Super-ego magnitude: {analysis['superego_magnitude']:.3f}")
    print()
    
    print("Network Architecture:")
    print(network)


if __name__ == "__main__":
    demonstrate_network()