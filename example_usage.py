"""
Example usage of the Super-Ego-Id Neural Network

This script demonstrates various aspects of the psychologically-inspired neural network,
including training scenarios and component analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from super_ego_id_network import SuperEgoIdNetwork, create_sample_data


def create_psychological_dataset(n_samples: int = 1000):
    """
    Create a synthetic dataset representing different psychological scenarios.
    
    The dataset includes scenarios that might trigger different psychological responses:
    - Immediate gratification scenarios (id-dominant)
    - Balanced decision scenarios (ego-dominant)  
    - Moral constraint scenarios (super-ego dominant)
    """
    
    # Create different scenario types
    scenarios = []
    labels = []
    
    # Id-dominant scenarios (immediate gratification)
    id_scenarios = torch.randn(n_samples // 3, 20) + torch.tensor([2.0] * 10 + [-1.0] * 10)
    scenarios.append(id_scenarios)
    labels.extend([0] * (n_samples // 3))  # Label 0 for id-dominant
    
    # Ego-dominant scenarios (balanced decisions)
    ego_scenarios = torch.randn(n_samples // 3, 20) * 0.5
    scenarios.append(ego_scenarios)
    labels.extend([1] * (n_samples // 3))  # Label 1 for ego-dominant
    
    # Super-ego dominant scenarios (moral constraints)
    superego_scenarios = torch.randn(n_samples // 3, 20) + torch.tensor([-2.0] * 10 + [2.0] * 10)
    scenarios.append(superego_scenarios)
    labels.extend([2] * (n_samples // 3))  # Label 2 for super-ego dominant
    
    # Combine all scenarios
    X = torch.cat(scenarios, dim=0)
    y = torch.tensor(labels, dtype=torch.long)
    
    # Shuffle the dataset
    indices = torch.randperm(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def train_network(network, X_train, y_train, epochs=100):
    """Train the Super-Ego-Id network on psychological scenarios."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    
    losses = []
    
    print("Training Super-Ego-Id Network...")
    print("-" * 40)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        output, components = network(X_train)
        loss = criterion(output, y_train)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return losses


def analyze_component_responses(network, X_test, y_test):
    """Analyze how different components respond to different scenario types."""
    
    network.eval()
    
    component_responses = {'id': [], 'ego': [], 'superego': []}
    scenario_types = ['Id-dominant', 'Ego-dominant', 'Super-ego dominant']
    
    with torch.no_grad():
        for scenario_type in range(3):
            # Get samples for this scenario type
            mask = y_test == scenario_type
            if mask.sum() == 0:
                continue
                
            scenario_samples = X_test[mask]
            
            # Get component analysis
            analysis = network.get_component_analysis(scenario_samples)
            
            component_responses['id'].append(analysis['id_dominance'])
            component_responses['ego'].append(analysis['ego_dominance'])
            component_responses['superego'].append(analysis['superego_dominance'])
            
            print(f"\n{scenario_types[scenario_type]} Scenarios:")
            print(f"  Id dominance: {analysis['id_dominance']:.3f}")
            print(f"  Ego dominance: {analysis['ego_dominance']:.3f}")
            print(f"  Super-ego dominance: {analysis['superego_dominance']:.3f}")
    
    return component_responses


def visualize_component_responses(component_responses):
    """Visualize how different components respond to different scenarios."""
    
    scenario_types = ['Id-dominant', 'Ego-dominant', 'Super-ego dominant']
    components = ['Id', 'Ego', 'Super-ego']
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(scenario_types))
    width = 0.25
    
    # Plot bars for each component
    for i, component in enumerate(components):
        values = component_responses[component.lower().replace('-', '')]
        ax.bar(x + i * width, values, width, label=component, alpha=0.8)
    
    ax.set_xlabel('Scenario Types')
    ax.set_ylabel('Component Dominance')
    ax.set_title('Component Responses to Different Psychological Scenarios')
    ax.set_xticks(x + width)
    ax.set_xticklabels(scenario_types)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/runner/work/Not-trained-Neural-Networks-Notes/Not-trained-Neural-Networks-Notes/component_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


def demonstrate_attention_mechanism(network, sample_input):
    """Demonstrate the attention mechanism in the network."""
    
    network.eval()
    
    with torch.no_grad():
        output, components = network(sample_input)
        attention_weights = components['attention_weights']
        
        print("\nAttention Mechanism Analysis:")
        print("-" * 40)
        
        # Average attention weights across batches and heads
        avg_attention = torch.mean(attention_weights, dim=(0, 1))
        
        print(f"Average attention to Id: {avg_attention[0].item():.3f}")
        print(f"Average attention to Ego: {avg_attention[1].item():.3f}")
        print(f"Average attention to Super-ego: {avg_attention[2].item():.3f}")
        
        return avg_attention


def main():
    """Main demonstration function."""
    
    print("Super-Ego-Id Neural Network - Complete Example")
    print("=" * 50)
    
    # Create network
    input_dim = 20
    hidden_dim = 64
    output_dim = 3  # Three psychological scenario types
    
    network = SuperEgoIdNetwork(input_dim, hidden_dim, output_dim)
    
    # Create psychological dataset
    print("\nCreating psychological scenarios dataset...")
    X, y = create_psychological_dataset(n_samples=1200)
    
    # Split into train/test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train the network
    losses = train_network(network, X_train, y_train, epochs=100)
    
    # Analyze component responses
    print("\nAnalyzing component responses to different scenarios...")
    component_responses = analyze_component_responses(network, X_test, y_test)
    
    # Visualize results
    print("\nCreating visualization...")
    try:
        visualize_component_responses(component_responses)
        print("Visualization saved as 'component_analysis.png'")
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # Demonstrate attention mechanism
    sample_input = create_sample_data(batch_size=10, input_dim=input_dim)
    demonstrate_attention_mechanism(network, sample_input)
    
    # Final network analysis
    print("\nFinal Network Architecture Summary:")
    print("-" * 40)
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\nComponent breakdown:")
    id_params = sum(p.numel() for p in network.id_network.parameters())
    ego_params = sum(p.numel() for p in network.ego_network.parameters())
    superego_params = sum(p.numel() for p in network.superego_network.parameters())
    
    print(f"Id network parameters: {id_params:,}")
    print(f"Ego network parameters: {ego_params:,}")
    print(f"Super-ego network parameters: {superego_params:,}")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()