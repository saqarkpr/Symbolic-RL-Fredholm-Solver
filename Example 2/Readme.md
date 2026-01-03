# 🧩 Example 2 – Oscillatory Fredholm Integral Equation (No Exact Solution)

This example tests the performance of the **Symbolic DQN Solver** on a smooth but *oscillatory* Fredholm integral equation of the second kind.  
Unlike Example 1, there is **no closed-form analytical solution**, so convergence behavior and residual error are used to evaluate performance.

---

## 📘 Problem Definition

$u(x) = f(x) + \lambda \int_0^1 K(x,t)\,u(t)\,dt, \quad 0 \le x \le 1,$

where

$K(x,t) = \sin(x - t), \quad f(x) = \sin(\pi x), \quad \lambda = 0.8.$

This equation produces oscillatory behavior in \($u(x)$\), making it a more challenging benchmark for symbolic learning systems.

---

## ⚙️ Model Description

- **Method:** Symbolic DQN Solver  
- **Goal:** Discover a symbolic function \( $u(x)$ \) minimizing the residual norm of the integral equation.  
- **Solver:** LBFGS optimizer for local fine-tuning of coefficients.  
- **Integration:** Gaussian–Legendre quadrature with 20 nodes.  
- **Episodes:** 15 000  
- **Max steps:** 100  
- **λ (Coupling coefficient):** 0.8  
- **Reward:** 
    $R = 1/(\text{MAE} + 10^{-5})$

---

## 🧠 Symbolic DQN Components

| Component | Description |
|------------|--------------|
| **Expression Grammar** | Expands symbolic expressions using context-free rules with operators {+, −, ×, ÷, sin, cos, exp, log}. |
| **Evaluator** | Builds differentiable computation graphs in PyTorch for each candidate expression. |
| **Solver Module** | Evaluates residuals \($u(x)-f(x)-\lambda\int K(x,t)u(t)dt$\) via Gaussian–Legendre integration. |
| **DQN Framework** | Guides symbolic search using experience replay and ε-greedy policy. |

---

## 📊 Evaluation Metrics

| Metric | Meaning |
|---------|----------|
| **RMS Residual** | Root mean square of the integral equation residual. |
| **MAE (if available)** | Mean absolute error (approximated numerically). |
| **Reward Trend** | Growth indicates improved symbolic fit. |

---

## 📂 File Information
- **File:** `Fredholm_SecKind_RL_DQN.py`
- **Log:** `RL_DQN_Results.txt`
- **Dependencies:** `torch`, `numpy`, `scipy`, `datetime`

---

## 🎯 Purpose
This case verifies that the Symbolic DQN Solver can model smooth oscillatory behaviors even without a known analytical solution—demonstrating its generalization ability beyond trivial linear equations.
