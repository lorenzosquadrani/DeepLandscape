import torch
import torch.nn as nn


class modelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8, 10)
        self.out = nn.Linear(10,2)
        
    def forward(self,x):
        x = F.relu(self.hidden1(x))
        x = self.out(x)
        return x
