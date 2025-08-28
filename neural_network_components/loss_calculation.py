import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaDesignLoss:
    def __init__(self, value_wt=1.0, policy_wt=1.0):
        self.value_wt = value_wt
        self.policy_wt = policy_wt
        self.mse_fn = nn.MSELoss()
        self.huber_fn = nn.SmoothL1Loss()  # learned this new thing 
    
    def compute_pipeline_loss(self, policy_out, value_out, cfd_vals, param_deltas):
        # val_loss = self.mse_fn(value_out.squeeze(), cfd_vals.float())
        val_loss = self.huber_fn(value_out.squeeze(), cfd_vals.float())
        
        pol_loss = self._compute_policy_loss(policy_out, param_deltas, cfd_vals)

        reg_term = self.compute_regularization_loss(policy_out)
        
        total_obj = (self.value_wt * val_loss +
                     self.policy_wt * pol_loss +
                     0.01 * reg_term)
        
        return total_obj, {
            'total': total_obj.item(),
            'value': val_loss.item(),
            'policy': pol_loss.item(),
            'regularization': reg_term.item()
        }

    def _compute_policy_loss(self, pol_out, deltas, cfd_vals):
        norm_deltas = torch.tanh(deltas / 10.0)
        adv_factors = torch.sigmoid(cfd_vals / 50.0)

        pol_loss = -torch.mean(pol_out * norm_deltas.unsqueeze(-1) * adv_factors.unsqueeze(-1))
        return pol_loss
    
    def compute_regularization_loss(self, pol_out):
        l2_penalty = torch.mean(torch.square(pol_out))
        
        if pol_out.size(0) > 1:
            smooth_penalty = torch.mean((pol_out[1:] - pol_out[:-1]) ** 2)
        else:
            smooth_penalty = torch.tensor(0.0, device=pol_out.device)

        return l2_penalty + 0.1 * smooth_penalty
    
    def cfd_reward_loss(self, val_preds, actual_cfd):
        norm_cfd = torch.sigmoid(actual_cfd / 100.0)
        return self.huber_fn(val_preds.squeeze(), norm_cfd)
    
    def compute_curriculum_loss(self, pol_out, val_out, cfd_vals, deltas, diff_factor=1.0):
        base_loss, loss_stats = self.compute_pipeline_loss(pol_out, val_out, cfd_vals, deltas)
        
        curr_loss = base_loss * diff_factor
        loss_stats['curriculum_factor'] = diff_factor
        
        return curr_loss, loss_stats
