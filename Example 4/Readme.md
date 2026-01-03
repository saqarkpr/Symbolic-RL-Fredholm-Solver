# 🧩 Example 4 – Stability Test under Large Coupling (λ > 1)

This example investigates the behavior of the **Symbolic DQN Solver** when solving a **Fredholm integral equation of the second kind** with a **strong coupling coefficient** (λ = 1.2).  
Such configurations often lead to ill-conditioned systems and provide an effective **stability and robustness test** for symbolic solvers.

---

## 📘 Problem Definition

$u(x) = f(x) + \lambda \int_0^1 K(x,t)\,u(t)\,dt, \quad 0 \le x \le 1,$

where  

$K(x,t) = e^{-(x-t)^2}, \quad f(x) = \sin(\pi x), \quad \lambda = 1.2.$

This setup amplifies the feedback between \($u(x)$\) and its integral term, testing whether the symbolic framework can maintain convergence and stability in the presence of strong coupling.

---

## ⚙️ Model Description

- **Method:** Symbolic DQN Solver  
- **Coupling coefficient:** λ = 1.2 (greater than unity)  
- **Integration:** Gaussian–Legendre quadrature with 25 nodes  
- **Episodes:** 60 000  
- **Epochs per model:** 60 (LBFGS optimization)  
- **Reward Function:**  
  
  $R = \frac{1}{\text{MAE} + 10^{-5}}$
  
- **Network:** Deep Q-Network with 2 hidden layers of 128 neurons each  
- **State size:** 30 (sequence of symbolic construction steps)  
- **Action space:** {0, 1, 2, 3, 4} for production-rule expansion  

---

## 🧠 Symbolic DQN Framework

| Component | Description |
|------------|--------------|
| **Grammar Expansion** | Generates symbolic forms using CFG-based rules. |
| **Residual Evaluation** | Computes \($u(x)-f(x)-\lambda\int K(x,t)u(t)dt$\) over collocation points. |
| **Reinforcement Learning Loop** | Learns to balance exploration (expression diversity) and exploitation (symbolic refinement). |
| **Optimizer** | LBFGS for coefficient refinement, Adam for DQN weight updates. |

---

## 📊 Evaluation Metrics

| Metric | Description |
|---------|-------------|
| **RMS Residual** | Root mean square of the integral residual over 25 collocation nodes. |
| **Convergence Rate** | Episodes required for residual < 1e−2. |
| **Stability** | Ability to converge despite λ > 1 feedback gain. |

---

## 📂 File Information
- **File:** `Fredholm_SecKind_RL_DQN.py`  
- **Log File:** `RL_DQN_Results.txt`  
- **Dependencies:** `torch`, `numpy`, `scipy`, `datetime`

---

## 🎯 Purpose
This example validates the **stability and robustness** of the Symbolic DQN Solver under strong feedback (λ > 1), demonstrating its applicability to stiff or ill-conditioned Fredholm systems where traditional PINN or collocation solvers tend to diverge.
