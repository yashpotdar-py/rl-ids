import os
import tempfile
import pytest
import torch
from rl_ids.modeling.q_network import QNetwork

import torch.nn as nn


@pytest.fixture
def sample_input():
    """Create a sample input tensor for testing."""
    return torch.rand(32, 10)  # Batch size 32, input dimension 10


class TestQNetwork:
    def test_init_single_hidden_layer(self):
        """Test QNetwork initialization with a single hidden layer."""
        network = QNetwork(input_dim=10, hidden_dims=64, output_dim=4)
        
        # Check structure
        assert isinstance(network.net, nn.Sequential)
        assert isinstance(network.net[0], nn.Linear)
        assert network.net[0].in_features == 10
        assert network.net[0].out_features == 64
        assert isinstance(network.net[-1], nn.Linear)
        assert network.net[-1].out_features == 4
        
    def test_init_multiple_hidden_layers(self):
        """Test QNetwork initialization with multiple hidden layers."""
        network = QNetwork(input_dim=10, hidden_dims=[64, 32], output_dim=4)
        
        # Find the linear layers and check their dimensions
        linear_layers = [m for m in network.net if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 3  # Input->64, 64->32, 32->Output
        assert linear_layers[0].in_features == 10
        assert linear_layers[0].out_features == 64
        assert linear_layers[1].in_features == 64
        assert linear_layers[1].out_features == 32
        assert linear_layers[2].in_features == 32
        assert linear_layers[2].out_features == 4
        
    def test_layer_norm_config(self):
        """Test QNetwork with and without layer normalization."""
        network_with_norm = QNetwork(input_dim=10, hidden_dims=64, output_dim=4, use_layer_norm=True)
        network_without_norm = QNetwork(input_dim=10, hidden_dims=64, output_dim=4, use_layer_norm=False)
        
        # Count LayerNorm instances
        norm_layers_with = sum(1 for m in network_with_norm.net if isinstance(m, nn.LayerNorm))
        norm_layers_without = sum(1 for m in network_without_norm.net if isinstance(m, nn.LayerNorm))
        
        assert norm_layers_with > 0
        assert norm_layers_without == 0
        
    def test_dropout_config(self):
        """Test QNetwork with and without dropout."""
        network_with_dropout = QNetwork(input_dim=10, hidden_dims=64, output_dim=4, dropout_rate=0.5)
        network_without_dropout = QNetwork(input_dim=10, hidden_dims=64, output_dim=4, dropout_rate=0.0)
        
        # Count Dropout instances
        dropout_layers_with = sum(1 for m in network_with_dropout.net if isinstance(m, nn.Dropout))
        dropout_layers_without = sum(1 for m in network_without_dropout.net if isinstance(m, nn.Dropout))
        
        assert dropout_layers_with > 0
        assert dropout_layers_without == 0
    
    def test_forward_pass(self, sample_input):
        """Test forward pass returns expected shape and type."""
        network = QNetwork(input_dim=10, hidden_dims=64, output_dim=4)
        output = network(sample_input)
        
        assert output.shape == (32, 4)  # Batch size 32, output dimension 4
        assert output.dtype == sample_input.dtype
    
    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        network = QNetwork(input_dim=10, hidden_dims=64, output_dim=4)
        
        # Test with batch size 1
        single_input = torch.rand(1, 10)
        single_output = network(single_input)
        assert single_output.shape == (1, 4)
        
        # Test with batch size 100
        large_input = torch.rand(100, 10)
        large_output = network(large_input)
        assert large_output.shape == (100, 4)
    
    def test_weight_initialization(self):
        """Test that weights are initialized according to expectations."""
        network = QNetwork(input_dim=10, hidden_dims=64, output_dim=4)
        
        for module in network.modules():
            if isinstance(module, nn.Linear):
                # Check weights aren't all zeros or ones
                assert not torch.allclose(module.weight, torch.zeros_like(module.weight))
                assert not torch.allclose(module.weight, torch.ones_like(module.weight))
                
                # Check biases are initialized to zero
                if module.bias is not None:
                    assert torch.allclose(module.bias, torch.zeros_like(module.bias))
    
    def test_save_and_load(self, sample_input):
        """Test saving and loading the model preserves the weights and outputs."""
        original_network = QNetwork(input_dim=10, hidden_dims=[64, 32], output_dim=4)
        original_output = original_network(sample_input)
        
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            # Save the model
            original_network.save(tmp.name)
            
            # Load the model
            loaded_network = QNetwork.load(tmp.name, device=torch.device('cpu'))
            loaded_output = loaded_network(sample_input)
            
            # Check model structure
            assert isinstance(loaded_network, QNetwork)
            assert len([m for m in loaded_network.net if isinstance(m, nn.Linear)]) == 3
            
            # Check outputs match
            assert torch.allclose(original_output, loaded_output)
    
    def test_model_config_in_saved_file(self):
        """Test that saved model contains the expected configuration."""
        network = QNetwork(input_dim=10, hidden_dims=[64, 32], output_dim=4)
        
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            # Save the model
            network.save(tmp.name)
            
            # Load the checkpoint directly to check contents
            checkpoint = torch.load(tmp.name)
            
            assert 'state_dict' in checkpoint
            assert 'model_config' in checkpoint
            assert checkpoint['model_config']['input_dim'] == 10
            assert checkpoint['model_config']['output_dim'] == 4
            assert checkpoint['model_config']['architecture'] == [64, 32]
    
    def test_edge_case_small_network(self, sample_input):
        """Test with a very small network."""
        small_network = QNetwork(input_dim=10, hidden_dims=[2], output_dim=1)
        output = small_network(sample_input)
        assert output.shape == (32, 1)
    
    def test_edge_case_large_dimensions(self):
        """Test with large input and output dimensions."""
        large_input_dim = 1000
        large_output_dim = 500
        large_batch_size = 10
        
        large_network = QNetwork(
            input_dim=large_input_dim, 
            hidden_dims=[128], 
            output_dim=large_output_dim
        )
        
        large_input = torch.rand(large_batch_size, large_input_dim)
        output = large_network(large_input)
        assert output.shape == (large_batch_size, large_output_dim)
        
    def test_different_devices(self):
        """Test model works on different devices if available."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create network and move to device
        network = QNetwork(input_dim=10, hidden_dims=64, output_dim=4)
        network.to(device)
        
        # Create input on device
        input_tensor = torch.rand(32, 10, device=device)
        output = network(input_tensor)
        
        # Compare device type rather than the exact device object 
        # (which may have additional index on CUDA)
        assert output.device.type == device.type
        assert output.shape == (32, 4)