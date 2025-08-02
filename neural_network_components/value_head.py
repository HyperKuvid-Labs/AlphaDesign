import torch
import torch.nn as nn

#so here we are gonna define the neural network arch for the value head

# so it goes eomthing similar to the policy head: 
# d_size(p_size)->d_size*4->d_size*2->1

# and then the activations would be relu(for now, for simplicity, considering gelu and silu too), and all of these are fully connected layers(dense)

class ValueHead(nn.Module):
    def __init__(self, p_size):
        super(ValueHead, self).__init__()
        self.fc1 = nn.Linear(p_size, p_size*4)
        self.fc2 = nn.Linear(p_size*4, p_size*2)
        self.fc3 = nn.Linear(p_size*2, 1)  
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))    
        x = self.relu(self.fc2(x))    
        x = self.fc3(x)              
        return x