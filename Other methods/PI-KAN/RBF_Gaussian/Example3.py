import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from scipy.special import roots_legendre
from torch.autograd import Variable
from PIKAN import KAN
import datetime

def Solve(model):
    from scipy.special import roots_legendre

    # ---------------- تنظیمات پایه ----------------
    Sampeling = 20      # نقاط تصادف و کوادراتور
    Epochs = 50
    a, b = 0.0, 1.0
    lambda_ = 0.7       # λ مثال سوم

    # ---------------- تعریف f و K ----------------
    def f(x):
        return torch.sin(torch.pi * x)

    def K(x, t):
        # کرنل غیرهمگن صاف
        return torch.exp(-(x - t)**2) * (1 + 0.5 * t**2)

    # ---------------- آماده‌سازی نقاط ----------------
    x = torch.linspace(a, b, Sampeling, requires_grad=False).reshape(-1, 1)
    roots, weights = roots_legendre(Sampeling)
    roots = torch.from_numpy(roots).float()
    weights = torch.from_numpy(weights).float()
    t_nodes = (b - a) / 2 * roots + (a + b) / 2  # نگاشت [-1,1] به [a,b]

    optimizer = optim.LBFGS(list(model.parameters()), lr=0.1, max_iter=20, line_search_fn="strong_wolfe")

    # ---------------- حلقه آموزش ----------------
    def Closure():
        y_pred = model.forward(x)                                 # u(x)
        u_t = model.forward(t_nodes.reshape(-1, 1)).reshape(-1)   # u(t)
        Kmat = K(x, t_nodes)                                      # K(x,t)
        integrand = Kmat * u_t.unsqueeze(0)                       # (M,N)
        integral_vec = ((b - a) / 2.0) * (integrand @ weights)    # (M,)
        residual = y_pred.reshape(-1) - (f(x).reshape(-1) + lambda_ * integral_vec)
        loss = (residual ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        return loss

    for _ in range(Epochs):
        optimizer.step(Closure)

    # ---------------- ارزیابی نهایی ----------------
    with torch.no_grad():
        y_pred = model.forward(x).reshape(-1)
        u_t = model.forward(t_nodes.reshape(-1, 1)).reshape(-1)
        Kmat = K(x, t_nodes)
        integrand = Kmat * u_t.unsqueeze(0)
        integral_vec = ((b - a) / 2.0) * (integrand @ weights)
        residual = y_pred - (f(x).reshape(-1) + lambda_ * integral_vec)
        rms_res = torch.sqrt((residual ** 2).mean())
    return rms_res.float()


u =  KAN(in_dim=1, hidden_dim=10, out_dim=1, grid=5)
start_time = datetime.datetime.now()
res = Solve(u)
elapsed_time = datetime.datetime.now() - start_time
print(f"MAE/Res = {res}")
print(f"Execution time = {elapsed_time}")