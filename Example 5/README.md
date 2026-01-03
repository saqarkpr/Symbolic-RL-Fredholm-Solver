# Distributed RC Line - Reinforcement Learning Model

This repository contains an implementation of a symbolic–numeric model based on **Deep Q-Learning (DQN)** for solving integral equations of the **Fredholm type (second kind)**.  
The current focus is a simplified **distributed RC line model** in steady-state form, where the voltage at each point along the line depends not only on local excitation but also on the voltages at neighboring points through a nonlocal coupling kernel.

---

## 📘 Mathematical Model

The integral formulation is expressed as:

$V(x) = V_s(x) + \lambda \int_0^1 e^{-\alpha |x-t|}V(t)dt,$

where:
- $V(x)$: Voltage along the line  
- $V_s(x)$: Distributed voltage source  
- $\lambda$: Coupling coefficient (depends on $R$ and $C$)  
- $J(x,t) = e^{-\alpha |x-t|}$: Exponential spatial kernel modeling RC attenuation  
- $\alpha > 0$: Damping constant (e.g., $\alpha = 5$)  

This equation represents a **nonlocal steady-state model** of an RC transmission line, which captures resistive losses and capacitive coupling between infinitesimal segments of the line.

---

## ⚙️ Implementation Overview

The solver is designed to work with a grammar-based symbolic model trained via **Reinforcement Learning (RL)**.  
Each candidate symbolic expression $V(x)$ is generated, evaluated, and optimized using an integral residual–based fitness function:

```python
V(x) = model.forward(x)
residual = V(x) - (V_s(x) + λ * ∫ J(x,t)V(t)dt)
loss = mean(residual ** 2)
````

The integral is approximated using **Gauss–Legendre quadrature**, ensuring high numerical stability even for smooth exponential kernels.

Training of the symbolic expressions follows a standard DQN workflow:

1. Expression generation using CFG rules
2. Action evaluation through the residual-based loss
3. Reward computation as `reward = 1 / (mae + regularization)`
4. Experience replay and target-network updates
5. Iterative optimization with `LBFGS` inside the fitness function

---

## 🧩 Example Parameters

| Parameter | Description              | Typical Value |
| --------- | ------------------------ | ------------- |
| λ         | Coupling coefficient     | 0.6           |
| α         | Attenuation rate         | 5.0           |
| a,b       | Interval bounds          | 0.0 – 1.0     |
| Sampeling | Gauss–Legendre nodes     | 20            |
| Epochs    | LBFGS optimization steps | 50            |

---

## 📈 Physical Interpretation

In a distributed RC line, each infinitesimal section acts as a small RC network that interacts with nearby sections through capacitive effects.
The exponential kernel ($e^{-\alpha |x-t|}$) captures this spatial influence, decaying with distance — closer points contribute more strongly to the potential at ($x$).

This formulation is analogous to simplified models used in:

* Transmission line analysis under diffusive approximation
* Delay and attenuation modeling in on-chip interconnects
* One-dimensional diffusion-like circuits with memory effects

---

## 🧮 Numerical Stability

The integral operator with exponential kernel is **positive-definite and compact**, which ensures convergence of the Fredholm formulation.
The combination of Gauss–Legendre quadrature and `LBFGS` minimization produces smooth residuals, and the RL component adapts symbolic structures that minimize them iteratively.

---

## 📂 File Structure

```
├── src/
│   ├── grammar.py              # CFG-based expression generator
│   ├── dqn_agent.py            # DQN agent for symbolic exploration
│   ├── solver_rc.py            # Integral solver for the RC model
│   └── utils.py                # Reward, logging, and data utilities
│
├── results/
│   ├── logs/
│   └── RL_RC_Results.txt
│
└── README.md                   # This file
```

---

## 🔬 Current Status

* ✅ Baseline model tested on smooth kernels (`exp(-α|x-t|)` and `exp(-(x-t)^2)`)
* ✅ Stable reward progression observed for λ ≤ 0.8
* ⚙️ Ongoing experiments for λ > 1 (stability threshold test)
* 🧠 Future: comparison with classical Nyström and collocation methods

---

## 📎 Citation (for internal reference)

This setup is part of an MSc research project on **symbolic reinforcement learning for integral equations** supervised at **Shahid Beheshti University**.

---

*Last updated: November 2025*
