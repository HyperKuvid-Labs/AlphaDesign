import torch
import torch.nn as nn
from .forward_pass import NeuralNetworkForwardPass

class NetworkInitializer:

    @staticmethod
    def _apply_he_initialization(network):
        for module in network.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias, 0)
    
    @staticmethod
    def setup_network(design_param_count, device="cpu"):
        network = NeuralNetworkForwardPass(design_param_count)
        network.to(device)
        
        NetworkInitializer._apply_he_initialization(network)

        param_count = sum(p.numel() for p in network.parameters() if p.requires_grad)

        return network, param_count
    
    @staticmethod
    def get_network_info(network):
        total_params = sum(p.numel() for p in network.parameters())
        trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'policy_head_params': sum(p.numel() for p in network.policy_head.parameters()),
            'value_head_params': sum(p.numel() for p in network.value_head.parameters())
        }