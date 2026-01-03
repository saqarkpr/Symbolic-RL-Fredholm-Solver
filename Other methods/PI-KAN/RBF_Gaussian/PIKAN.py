import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from scipy.special import roots_legendre
from torch.autograd import Variable

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid=5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid = grid
        # ضرایب ترکیب توابع پایه
        self.weights = nn.Parameter(torch.randn(out_features, in_features, grid))
        self.bias = nn.Parameter(torch.zeros(out_features))
        # گره‌های شبکه‌ی KAN
        self.centers = nn.Parameter(torch.linspace(-1, 1, grid))
        self.scale = nn.Parameter(torch.ones(1))
        
    def basis(self, x):
        # توابع پایه (مثل Gaussian یا sinusoidal)
        return torch.exp(-self.scale * (x.unsqueeze(-1) - self.centers) ** 2)

    def forward(self, x):
        Φ = self.basis(x)                              # (batch, in_features, grid)
        out = torch.einsum('bng,ong->bo', Φ, self.weights)
        return out + self.bias

class KAN(nn.Module):
    def __init__(self, in_dim, hidden_dim=10, out_dim=1, grid=5):
        super().__init__()
        self.layer1 = KANLayer(in_dim, hidden_dim, grid)
        self.layer2 = KANLayer(hidden_dim, hidden_dim, grid)
        self.layer3 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        return self.layer3(x)
