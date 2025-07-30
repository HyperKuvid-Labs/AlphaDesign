import torch
import torch.nn as nn
import numpy as np

class ParamterTweaker:
    def __init__(self, param_bounds=None, mod_scale=0.1):
        self.param_bounds = param_bounds
        self.mod_scale = mod_scale

    def apply_neural_tweaks(self, current_params, policy_output, exploration=True):
        mod_d = torch.tanh(policy_output)

        param_changes = mod_d * self.mod_scale
        modified_params = current_params + param_changes

        if exploration:
            noise = torch.randn_like(modified_params) * 0.01
            modified_params += noise


        if self.param_bounds is not None:
            modified_params = torch.clamp(modified_params, self.param_bounds['min'], self.param_bounds['max'])

        return modified_params
    
    def genetic_to_neural_params(self, genetic_population):
        return torch.tensor(genetic_population, dtype=torch.float32)
    

    def neural_to_cfd(self, neural_params):
        return neural_params.cpu().numpy()
