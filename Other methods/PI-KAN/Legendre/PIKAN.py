import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from scipy.special import roots_legendre
from torch.autograd import Variable
import torch
import torch.nn as nn

# ---------------------------------------------
# پایه‌های لژاندر متعامد تا درجه n
# ---------------------------------------------
class LegendreBasis(nn.Module):
    def __init__(self, degree=5):
        super().__init__()
        self.degree = degree
    
    def forward(self, x):
        # فرض: x ∈ [-1, 1]
        x = x.unsqueeze(-1)
        P = [torch.ones_like(x), x]  # P0, P1
        for n in range(2, self.degree):
            Pn = ((2*n - 1)*x*P[-1] - (n - 1)*P[-2]) / n
            P.append(Pn)
        Φ = torch.cat(P[:self.degree], dim=-1)  # (batch, degree)
        return Φ


# ---------------------------------------------
# لایه‌ی KAN بر پایه‌ی توابع لژاندر
# ---------------------------------------------
class LegendreKANLayer(nn.Module):
    def __init__(self, in_features, out_features, degree=6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree

        self.weights = nn.Parameter(torch.randn(out_features, in_features, degree))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.basis = LegendreBasis(degree)
    
    def forward(self, x):
        # x: (batch, in_features)
        Φ = self.basis(x)                     # (batch, in_features, degree)
        out = torch.einsum('bnd,ond->bo', Φ, self.weights)
        return out + self.bias


# ---------------------------------------------
# شبکه‌ی نهایی KAN بر پایه‌ی لژاندر
# ---------------------------------------------
class LegendreKAN(nn.Module):
    def __init__(self, in_dim, hidden_dim=10, out_dim=1, degree=6):
        super().__init__()
        self.layer1 = LegendreKANLayer(in_dim, hidden_dim, degree)
        self.layer2 = LegendreKANLayer(hidden_dim, hidden_dim, degree)
        self.layer3 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.Tanh()

    def forward(self, x):
        # نرمال‌سازی ورودی به بازه‌ی [-1,1]
        x = 2 * (x - x.min()) / (x.max() - x.min()) - 1
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        return self.layer3(x)
