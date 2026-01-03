# 🧩 Example 1 – Fredholm Integral Equation with Exact Solution

This example serves as the **baseline validation case** for the proposed **Symbolic DQN Solver**.  
The equation under consideration is a **linear Fredholm integral equation of the second kind** with a known analytical solution.

---

## 📘 Problem Definition

The equation is defined as:

$u(x) = f(x) + \lambda \int_0^1 K(x,t)\,u(t)\,dt, \quad 0 \le x \le 1,$

where  
$K(x,t) = e^t, \quad f(x) = x e, \quad \lambda = 1.$  

The analytical solution is:
$u(x) = x.$

This benchmark is used to verify that the Symbolic DQN agent can rediscover the correct closed-form expression.

---

## ⚙️ Model Description

- **Method:** Symbolic DQN Solver  
- **Goal:** Discover a symbolic closed-form expression \($u(x)$\) that minimizes the residual of the integral equation.  
- **Reward Function:**  
  $
  R = \frac{1}{\text{MAE} + 10^{-5}}
  $
- **Optimization:** Adam optimizer for Q-network, LBFGS for coefficient tuning in the symbolic expression.  
- **Episodes:** 1500  
- **Max steps per episode:** 30  
- **State size:** 30  
- **Action space:** `[0, 1, 2, 3, 4]` corresponding to expression-building actions.

---

## 🧠 Symbolic Framework Components

| Component | Description |
|------------|--------------|
| **Expression Generator** | Expands symbolic trees using grammar-based rules. |
| **Model Evaluator** | Builds a PyTorch computational graph for the generated expression. |
| **Solver** | Minimizes residuals of the Fredholm equation using LBFGS. |
| **Reward Signal** | Based on inverse MAE between predicted and exact \($u(x)$\). |

---

## 📊 Evaluation Metrics

| Metric | Description |
|---------|-------------|
| **MAE** | Mean absolute error against analytical solution \($u(x)=x$\). |
| **Reward** | Inverse of error, used to guide DQN exploration. |
| **Convergence** | Achieved when MAE < 0.01. |

---

## 📂 File Information
- **File:** `Fredholm_SecKind_RL_DQN.py`
- **Log Output:** `RL_DQN_Results.txt`
- **Dependencies:** `torch`, `numpy`, `scipy`, `datetime`

---

## 🎯 Purpose
This example provides a reference for evaluating the correctness of the Symbolic DQN Solver before applying it to more complex Fredholm equations without analytical solutions.
