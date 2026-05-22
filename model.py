import torch
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, num_features, projection_dim, inner_layers, outputs):
        super(NN, self).__init__()
        
        self.projection = nn.Linear(num_features, projection_dim)
        
        modules = []
        current_dim = projection_dim
        
        for next_dim in inner_layers:
            modules.append(nn.Linear(current_dim, next_dim))
            modules.append(nn.ReLU())
            current_dim = next_dim
            
        modules.append(nn.Linear(current_dim, outputs))
        
        self.internal_network = nn.Sequential(*modules)

    def forward(self, x_input):
        x_projected = self.projection(x_input)
        result = self.internal_network(x_projected)
        return result

heuristic = NN(324, 128, [256, 128], 21)
policy = NN(324, 128, [256, 128], 18)