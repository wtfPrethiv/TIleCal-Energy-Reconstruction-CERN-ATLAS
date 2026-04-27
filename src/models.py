import torch.nn as nn

# Linear Regression model construction
class LinearRegression(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        
        # linear layer: (in_dim features) -> (1 output)
        # internally stores weight matrix and bias term
        self.linear = nn.Linear(in_dim, 1)
        
    # forward pass    
    def forward(self, X):
        # apply linear transformation
        # output = XW + b
        return self.linear(X)