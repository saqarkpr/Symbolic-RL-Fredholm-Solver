import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from scipy.special import roots_legendre
from torch.autograd import Variable
from PIKAN import LegendreKAN
import datetime

def Solve(model):

    Sampeling = 20
    Epochs = 50
    a,b=0,1

    
    def Exact(x):
        return x
    
    def K(y, x):
        return torch.exp(y)

    def Fredholm_integral_eq(y, x):
        g = x * torch.e**1
        roots, weights = roots_legendre(Sampeling)
        weights = torch.tensor(weights, dtype=torch.float)
        int_term = (b - a) / 2 * K(y, x).flatten().dot(weights)
        return y + (x * int_term - g)

    
    x = torch.linspace(a, b, Sampeling, requires_grad=True).reshape(-1, 1)

    optimizer = optim.LBFGS(list(model.parameters()), lr=0.1)

    def Closure():
        y = model.forward(x)
        # Compute the integral terms using trapezoidal rule
        integral_term = Fredholm_integral_eq(y, x)
        # Define the residual based on the integral equation
        residual = integral_term
        #initial = y[0] - 0
        loss = (residual**2).mean() #+ initial**2

        optimizer.zero_grad()
        loss.backward()
        return loss
    
    def Validation():
        x_test = torch.linspace(a,b, 33).reshape(-1, 1)
        exact = Exact(x_test)
        predict = model.forward(x_test).detach().numpy()
        error = exact - predict

        MAE = torch.abs(error).mean()
        return MAE
     
    for i in range(Epochs):
        optimizer.step(Closure)
        
    return torch.tensor(Validation()).float()

u =  LegendreKAN(in_dim=1, hidden_dim=5, out_dim=1,degree=4)
start_time = datetime.datetime.now()
res = Solve(u)
elapsed_time = datetime.datetime.now() - start_time
print(f"MAE/Res = {res}")
print(f"Execution time = {elapsed_time}")