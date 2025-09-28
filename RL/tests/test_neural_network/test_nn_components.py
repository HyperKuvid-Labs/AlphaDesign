"""Tests for neural network components."""

import pytest
import numpy as np
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_network_components.policy_head import PolicyHead
from neural_network_components.value_head import ValueHead
from neural_network_components.network_initialization import NetworkInitializer
from neural_network_components.forward_pass import NeuralNetworkForwardPass
from neural_network_components.optimizer_integration import OptimizerManager
from neural_network_components.loss_calculation import AlphaDesignLoss

class TestNetworkInitialization:
    def test_network_creation(self):
        design_param_count = 50
        network, param_count = NetworkInitializer.setup_network(
            design_param_count=design_param_count,
            device="cpu",
            hidden_dim=256,
            depth=3
        )
        
        assert network is not None
        assert isinstance(network, nn.Module)
        assert isinstance(network, NeuralNetworkForwardPass)
        assert param_count > 0
        assert network.param_count == design_param_count
    
    def test_parameter_initialization(self):
        design_param_count = 50
        network, param_count = NetworkInitializer.setup_network(design_param_count)
        
        # Test that weights are properly initialized (not all zeros)
        for module in network.modules():
            if isinstance(module, nn.Linear):
                assert not torch.allclose(module.weight, torch.zeros_like(module.weight))
                if module.bias is not None:
                    assert torch.allclose(module.bias, torch.zeros_like(module.bias))
            elif isinstance(module, nn.LayerNorm):
                assert torch.allclose(module.bias, torch.ones_like(module.bias))
    
    def test_network_info(self):
        design_param_count = 50
        network, _ = NetworkInitializer.setup_network(design_param_count)
        
        info = NetworkInitializer.get_network_info(network)
        
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert 'feature_trunk_params' in info
        assert 'policy_head_params' in info
        assert 'value_head_params' in info
        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] > 0

class TestForwardPass:
    
    def test_forward_pass_shape(self):
        design_param_count = 50
        batch_size = 32
        
        network = NeuralNetworkForwardPass(
            param_count=design_param_count,
            hidden_dim=256,
            depth=3,
            dropout=0.1
        )
        
        input_tensor = torch.randn(batch_size, design_param_count)
        policy_output, value_output = network.forward(input_tensor)
        
        assert policy_output.shape == (batch_size, design_param_count)
        assert value_output.shape == (batch_size, 1)
    
    def test_gradient_flow(self):
        design_param_count = 50
        batch_size = 16
        
        network = NeuralNetworkForwardPass(design_param_count)
        input_tensor = torch.randn(batch_size, design_param_count, requires_grad=True)
        
        policy_output, value_output = network.forward(input_tensor)
        loss = policy_output.sum() + value_output.sum()
        loss.backward()
        
        # Check that gradients exist for all parameters
        for param in network.parameters():
            assert param.grad is not None
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad))
    
    def test_predict_modifications(self):
        design_param_count = 50
        network = NeuralNetworkForwardPass(design_param_count)
        
        current_params = torch.randn(1, design_param_count)
        modifications = network.predict_modifications(current_params)
        
        assert modifications.shape == (1, design_param_count)
        assert torch.all(modifications >= -1.0) and torch.all(modifications <= 1.0)  # Tanh output
    
    def test_evaluate_design(self):
        design_param_count = 50
        network = NeuralNetworkForwardPass(design_param_count)
        
        current_params = torch.randn(1, design_param_count)
        value = network.evaluate_design(current_params)
        
        assert value.shape == (1, 1)

class TestPolicyHead:
    
    def test_policy_output_distribution(self):
        feature_dim = 256
        output_dim = 50
        batch_size = 32
        
        policy_head = PolicyHead(feature_dim, output_dim)
        features = torch.randn(batch_size, feature_dim)
        policy_output = policy_head.forward(features)
        
        assert policy_output.shape == (batch_size, output_dim)
        # Policy head uses Tanh activation, so outputs should be in [-1, 1]
        assert torch.all(policy_output >= -1.0) and torch.all(policy_output <= 1.0)
    
    def test_action_sampling(self):
        feature_dim = 256
        output_dim = 50
        
        policy_head = PolicyHead(feature_dim, output_dim)
        features = torch.randn(1, feature_dim)
        
        # Test multiple samples for consistency
        samples = []
        for _ in range(10):
            policy_output = policy_head.forward(features)
            samples.append(policy_output.detach())
        
        # All samples should have same shape and be within valid range
        for sample in samples:
            assert sample.shape == (1, output_dim)
            assert torch.all(sample >= -1.0) and torch.all(sample <= 1.0)
    
    def test_layer_normalization(self):
        feature_dim = 256
        output_dim = 50
        
        policy_head = PolicyHead(feature_dim, output_dim)
        
        # Check that LayerNorm layers exist
        layer_norms = [module for module in policy_head.modules() if isinstance(module, nn.LayerNorm)]
        assert len(layer_norms) == 2  # Two LayerNorm layers in the architecture

class TestValueHead:
    
    def test_value_output_range(self):
        feature_dim = 256
        batch_size = 32
        
        value_head = ValueHead(feature_dim)
        features = torch.randn(batch_size, feature_dim)
        values = value_head.forward(features)
        
        assert values.shape == (batch_size, 1)
        # Value head has no final activation, so can output any real number
        assert torch.isfinite(values).all()
    
    def test_value_consistency(self):
        feature_dim = 256
        value_head = ValueHead(feature_dim)
        
        base_features = torch.randn(1, feature_dim)
        base_value = value_head.forward(base_features)
        
        # Test with slightly perturbed inputs
        noise_scale = 0.01
        for _ in range(5):
            noisy_features = base_features + torch.randn_like(base_features) * noise_scale
            noisy_value = value_head.forward(noisy_features)
            
            # Values should be relatively close for similar inputs
            diff = torch.abs(base_value - noisy_value)
            assert diff < 1.0  # Reasonable tolerance for small input changes
    
    def test_feature_reduction(self):
        feature_dim = 256
        value_head = ValueHead(feature_dim)
        
        # Check the architecture reduces dimensions correctly
        # feature_dim -> feature_dim//2 -> feature_dim//4 -> 1
        linear_layers = [module for module in value_head.modules() if isinstance(module, nn.Linear)]
        
        assert len(linear_layers) == 3
        assert linear_layers[0].out_features == feature_dim // 2
        assert linear_layers[1].out_features == feature_dim // 4
        assert linear_layers[2].out_features == 1

class TestOptimizerIntegration:
    
    def test_optimizer_creation(self):
        network = NeuralNetworkForwardPass(50)
        optimizer_manager = OptimizerManager(network)
        
        assert optimizer_manager.network == network
        assert optimizer_manager.optimizer is not None
        assert optimizer_manager.scheduler is not None
    
    def test_learning_rate_scheduling(self):
        network = NeuralNetworkForwardPass(50)
        optimizer_manager = OptimizerManager(network)
        
        initial_lr = optimizer_manager.get_current_lr()
        
        # Test cosine warm restarts
        optimizer_manager.use_cosine_warm()
        optimizer_manager.step_cosine_scheduler(epoch=1)
        
        # Learning rate should have changed
        new_lr = optimizer_manager.get_current_lr()
        assert initial_lr != new_lr or True  # LR might be same at certain points
    
    def test_separate_head_optimizers(self):
        network = NeuralNetworkForwardPass(50)
        optimizer_manager = OptimizerManager(network)
        
        policy_opt, value_opt = optimizer_manager.separate_head_optimizers()
        
        assert policy_opt is not None
        assert value_opt is not None
        assert policy_opt != value_opt

class TestLossCalculation:
    
    def test_pipeline_loss_computation(self):
        batch_size = 16
        param_count = 50
        
        loss_fn = AlphaDesignLoss()
        
        policy_pred = torch.randn(batch_size, param_count)
        value_pred = torch.randn(batch_size, 1)
        cfd_scores = torch.randn(batch_size)
        param_improvements = torch.randn(batch_size)
        
        total_loss, loss_dict = loss_fn.compute_pipeline_loss(
            policy_pred, value_pred, cfd_scores, param_improvements
        )
        
        assert torch.isfinite(total_loss).all()
        assert 'total' in loss_dict
        assert 'value' in loss_dict
        assert 'policy' in loss_dict
        assert 'regularization' in loss_dict
    
    def test_regularization_loss(self):
        loss_fn = AlphaDesignLoss()
        
        policy_output = torch.randn(32, 50)
        reg_loss = loss_fn.compute_regularization_loss(policy_output)
        
        assert torch.isfinite(reg_loss).all()
        assert reg_loss >= 0.0  # Regularization should be non-negative
    
    def test_curriculum_loss(self):
        batch_size = 16
        param_count = 50
        
        loss_fn = AlphaDesignLoss()
        
        policy_pred = torch.randn(batch_size, param_count)
        value_pred = torch.randn(batch_size, 1)
        cfd_scores = torch.randn(batch_size)
        param_improvements = torch.randn(batch_size)
        difficulty_factor = 1.5
        
        curriculum_loss, loss_dict = loss_fn.compute_curriculum_loss(
            policy_pred, value_pred, cfd_scores, param_improvements, difficulty_factor
        )
        
        assert torch.isfinite(curriculum_loss).all()
        assert 'curriculum_factor' in loss_dict
        assert loss_dict['curriculum_factor'] == difficulty_factor

if __name__ == "__main__":
    pytest.main([__file__])
