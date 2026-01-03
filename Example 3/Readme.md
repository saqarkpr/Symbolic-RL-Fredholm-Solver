# 🧩 Example 3 – Smooth Non-Homogeneous Fredholm Integral Equation

This example evaluates the **Symbolic DQN Solver** on a **non-homogeneous Fredholm integral equation of the second kind**.  
The kernel combines Gaussian decay and polynomial dependence on \(t\), producing a nonlinear integral coupling that tests the solver’s expressivity.

---

## 📘 Problem Definition

$u(x) = f(x) + \lambda \int_0^1 K(x,t)\,u(t)\,dt, \quad 0 \le x \le 1,$

with  
$K(x,t) = e^{-(x - t)^2}\big(1 + 0.5 t^2\big), \quad f(x) = \sin(\pi x), \quad \lambda = 0.7.$

No analytical solution is known for this kernel, but the equation’s structure yields a smooth, monotonic solution suitable for residual-based evaluation.

---

## ⚙️ Model Description

- **Method:** Symbolic DQN Solver  
- **Target:** Learn a symbolic closed-form expression minimizing the integral residual norm.  
- **Integration:** Gaussian–Legendre quadrature with 20 nodes.  
- **Optimization:** LBFGS (for coefficient fitting) + DQN (for symbolic structure search).  
- **Episodes:** 60 000  
- **Max steps:** 100  
- **λ (Coupling):** 0.7  
- **Reward:**  
  
  $R = \frac{1}{\text{MAE} + 10^{-5}}$

---

## 🧠 Symbolic DQN Architecture

| Module | Description |
|---------|-------------|
| **Grammar Generator** | Expands symbolic trees using operators {+, −, ×, ÷, sin, cos, exp, log}. |
| **CFGBasedModel** | Builds a PyTorch computation graph from symbolic expressions. |
| **Integral Solver** | Evaluates residual \( $u(x)-f(x)-\lambda\int K(x,t)u(t)dt$ \). |
| **Reinforcement Agent** | Uses ε-greedy exploration and experience replay to optimize symbolic forms. |

---

## 📊 Evaluation Metrics

| Metric | Description |
|---------|-------------|
| **RMS Residual** | Root-mean-square residual across collocation points. |
| **Reward Trend** | Indicates improvement of symbolic structure accuracy. |
| **Convergence** | Typically achieved after several tens of thousands of episodes. |

---

## 📂 File Information
- **File:** `Fredholm_SecKind_RL_DQN.py`
- **Log File:** `RL_DQN_Results.txt`
- **Dependencies:** `torch`, `numpy`, `scipy`, `datetime`

---

## 🎯 Purpose
This example demonstrates the ability of the Symbolic DQN Solver to handle **smooth, non-homogeneous kernels** where polynomial–exponential mixtures are necessary — a key step toward generalizing beyond simple or oscillatory Fredholm equations.
