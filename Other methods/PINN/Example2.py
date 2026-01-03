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

    # ---------------- config (مثل قبل) ----------------
    Sampeling = 20     # تعداد نقاط هم برای x و هم برای کوادراتور
    Epochs = 50
    a, b = 0.0, 1.0
    lambda_ = 0.8      # λ مثال دومت

    # ---------------- تعریف f و K مثال 2 ----------------
    def f(x):  # f(x) آزاد؛ ساختگی ولی نرم
        return torch.sin(torch.pi * x)

    def K(x, t):  # کرنل نوسانی بدون تکینگی
        # x: (M,1) یا (M,), t: (N,)
        return torch.sin(x - t)

    # ---------------- شبکه نقاط ----------------
    # نقاط تصادف (collocation) برای x
    x = torch.linspace(a, b, Sampeling, requires_grad=False).reshape(-1, 1)  # (M,1)
    # نقاط و وزن‌های گاوس-لژاندر برای انتگرال روی t
    roots, weights = roots_legendre(Sampeling)  # numpy arrays
    roots = torch.from_numpy(roots).float()     # در بازه‌ی [-1,1]
    weights = torch.from_numpy(weights).float() # (N,)
    # نگاشت به [a,b]:
    t_nodes = (b - a) / 2 * roots + (a + b) / 2  # (N,)

    optimizer = optim.LBFGS(list(model.parameters()), lr=0.1, max_iter=20, line_search_fn="strong_wolfe")

    def Closure():
        # u(x) روی نقاط تصادف
        y_pred = model.forward(x)                       # (M,1)
        # u(t) روی نقاط کوادراتور؛ باید گرادیان‌پذیر بماند
        u_t = model.forward(t_nodes.reshape(-1, 1)).reshape(-1)  # (N,)

        # ماتریس K(x_i, t_q): شکل (M,N)
        Kmat = K(x, t_nodes)                            # broadcasting → (M,N)

        # ∫ K(x,t) u(t) dt ≈ (b-a)/2 * Σ w_q K(x,t_q) u(t_q)
        integrand = Kmat * u_t.unsqueeze(0)             # (M,N)
        integral_vec = ((b - a) / 2.0) * (integrand @ weights)   # (M,)

        # باقیمانده: u(x) - [ f(x) + λ ∫ K u ]
        residual = y_pred.reshape(-1) - (f(x).reshape(-1) + lambda_ * integral_vec)  # (M,)

        loss = (residual ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        return loss

    # چند بار LBFGS اجرا شود (مثل قبل)
    for _ in range(Epochs):
        optimizer.step(Closure)

    # بعد از آموزش، Residual RMS را برمی‌گردانیم (برای پاداش RL)
    with torch.no_grad():
        y_pred = model.forward(x).reshape(-1)
        u_t = model.forward(t_nodes.reshape(-1, 1)).reshape(-1)
        Kmat = K(x, t_nodes)
        integrand = Kmat * u_t.unsqueeze(0)
        integral_vec = ((b - a) / 2.0) * (integrand @ weights)
        residual = y_pred - (torch.sin(torch.pi * x).reshape(-1) + lambda_ * integral_vec)
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
