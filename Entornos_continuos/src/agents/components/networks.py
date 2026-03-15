import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    Red Neuronal dinámica para aproximar Q-values.
    Se adapta según el número de capas (num_layer) especificado.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layer=3):
        super(QNetwork, self).__init__()
        
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(num_layer - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        
        return self.fc_out(x)