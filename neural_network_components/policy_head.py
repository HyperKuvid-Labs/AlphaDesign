import torch
import torch.nn as nn

#so here we are ginna define the neural architecture of the policy head

# so architecture goes something like this(1st iteration)
# d_size(parameters-size) -> d_size*4 -> d_size*2 -> d_size

# so the part of increasing *x, is to make the neural network to learn more and explore more

# and then the activations would be relu(for now, for simplicity, considering gelu and silu too), and all of these are fully connected layers(dense)

class PolicyHead(nn.Module):
    def __init__(self, p_size):
        super(PolicyHead, self).__init__()
        d_size = p_size
        self.dense_layer1 = nn.Linear(d_size, d_size*4)
        self.dense_layer2 = nn.Linear(d_size*4, d_size*2)
        self.output_layer = nn.Linear(d_size*2, d_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.dense_layer1(x))
        x = self.relu(self.dense_layer2(x))
        x = self.output_layer(x)
        return x