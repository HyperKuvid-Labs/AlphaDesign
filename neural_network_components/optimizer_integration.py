import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

class OptimizerManager:
    def __init__(self, network, learning_rate=1e-3, weight_decay=1e-4):
        self.network = network
        self.lr = learning_rate
        self.weight_decay = weight_decay
        
        self.optimizer = optim.Adam(
            network.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            patience=50, 
            factor=0.5
        )
    
    def get_optimizer(self):
        return self.optimizer
    
    def update_learning_rate(self, cfd_loss):
        self.scheduler.step(cfd_loss)
    
    def get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    def separate_head_optimizers(self, policy_lr=1e-3, value_lr=1e-3):
        policy_optimizer = optim.Adam(
            self.network.policy_head.parameters(),
            lr=policy_lr,
            weight_decay=self.weight_decay
        )
        
        value_optimizer = optim.Adam(
            self.network.value_head.parameters(), 
            lr=value_lr,
            weight_decay=self.weight_decay
        )
        
        return policy_optimizer, value_optimizer
    
    def reset_optimizer_state(self):
        self.optimizer.state = {}
