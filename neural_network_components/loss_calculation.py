import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaDesignLoss:
    def __init__(self, value_weight=1.0, policy_weight=1.0):
        self.value_weight = value_weight
        self.policy_weight = policy_weight
        self.mse_loss = nn.MSELoss()
    
    def compute_pipeline_loss(self, policy_pred, value_pred, cfd_scores, param_improvements):
        value_loss = self.mse_loss(value_pred.squeeze(), cfd_scores.float())
        
        policy_loss = self._compute_policy_loss(policy_pred, param_improvements)
        
        total_loss = (self.value_weight * value_loss + 
                     self.policy_weight * policy_loss)
        
        return total_loss, {
            'total': total_loss.item(),
            'value': value_loss.item(), 
            'policy': policy_loss.item()
        }
    
    def _compute_policy_loss(self, policy_output, improvements):
        improvement_rewards = torch.where(improvements > 0, 1.0, -0.5)
        policy_loss = -torch.mean(policy_output * improvement_rewards.unsqueeze(-1))
        return policy_loss
    
    def cfd_reward_loss(self, value_predictions, actual_cfd_results):
        normalized_cfd = torch.sigmoid(actual_cfd_results) 
        return self.mse_loss(value_predictions.squeeze(), normalized_cfd)

