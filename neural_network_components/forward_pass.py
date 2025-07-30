import torch
import torch.nn as nn

from policy_head import PolicyHead
from value_head import ValueHead

class NeuralNetworkForwardPass(nn.Module):
    def __init__(self, param_count):
        super(NeuralNetworkForwardPass, self).__init__()
        self.param_count = param_count
        self.policy_head = PolicyHead(param_count)
        self.value_head = ValueHead(param_count)

    def forward(self, design_params):
        policy_output = self.policy_head(design_params)
        value_output = self.value_head(design_params)
        return policy_output, value_output
    
    def predict_modifications(self, current_params):
        return self.policy_head(current_params)
    
    def evaluate_design(self, current_params):
        return self.value_head(current_params)