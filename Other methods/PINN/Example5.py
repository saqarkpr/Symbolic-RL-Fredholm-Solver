import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from scipy.special import roots_legendre
from torch.autograd import Variable
import datetime

def Solve(model):
    from scipy.special import roots_legendre

    # ----- تنظیمات -----
    Sampeling = 20      # تعداد نقاط x و نقاط کوادراتور
    Epochs = 50
    a, b = 0.0, 1.0
    lambda_ = 0.6       # ضریب کوپلینگ
    alpha = 5.0         # نرخ فروکش نمایی در کرنل

    # ----- f(x) و کرنل J(x,t) -----
    def V_s(x):
        return torch.sin(torch.pi * x)  # تحریک توزیع‌شده روی خط

    def J(x, t):
        # x: (M,1), t: (N,) → خروجی: (M,N)
        return torch.exp(-alpha * torch.abs(x - t))

    # ----- آماده‌سازی نقاط -----
    x = torch.linspace(a, b, Sampeling, requires_grad=False).reshape(-1, 1)  # نقاط تصادف
    roots, weights = roots_legendre(Sampeling)
    roots = torch.from_numpy(roots).float()
    weights = torch.from_numpy(weights).float()
    t_nodes = (b - a) / 2 * roots + (a + b) / 2  # نگاشت [-1,1] → [a,b]

    optimizer = optim.LBFGS(list(model.parameters()), lr=0.1,
                            max_iter=20, line_search_fn="strong_wolfe")

    # ----- حلقه آموزش -----
    def Closure():
        Vx = model.forward(x)                                   # V(x)
        Vt = model.forward(t_nodes.reshape(-1, 1)).reshape(-1)  # V(t)
        Jmat = J(x, t_nodes)                                   # J(x,t)
        integrand = Jmat * Vt.unsqueeze(0)                     # (M,N)
        integral_vec = ((b - a) / 2.0) * (integrand @ weights) # (M,)

        residual = Vx.reshape(-1) - (V_s(x).reshape(-1) + lambda_ * integral_vec)
        loss = (residual ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        return loss

    for _ in range(Epochs):
        optimizer.step(Closure)

    # ----- ارزیابی نهایی -----
    with torch.no_grad():
        Vx = model.forward(x).reshape(-1)
        Vt = model.forward(t_nodes.reshape(-1, 1)).reshape(-1)
        Jmat = J(x, t_nodes)
        integrand = Jmat * Vt.unsqueeze(0)
        integral_vec = ((b - a) / 2.0) * (integrand @ weights)
        residual = Vx - (V_s(x).reshape(-1) + lambda_ * integral_vec)
        rms_res = torch.sqrt((residual ** 2).mean())

    return rms_res.float()

u = nn.Sequential(
    nn.Linear(1, 10),
    nn.Tanh(),
    nn.Linear(10, 10),    
    nn.Tanh(),
    nn.Linear(10, 1),
)
start_time = datetime.datetime.now()
res = Solve(u)
elapsed_time = datetime.datetime.now() - start_time
print(f"MAE/Res = {res}")
print(f"Execution time = {elapsed_time}")
