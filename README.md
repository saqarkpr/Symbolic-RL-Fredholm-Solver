# Symbolic Reinforcement Learning for Solving Second-Kind Fredholm Integral Equations

**Author:** Saghar Khalilpour  
**Supervisor:** Dr. Kourosh Parand  
**Affiliation:** M.Sc. Thesis — Shahid Beheshti University, Department of Computer Science, 2025  
**Thesis Title:** Solving Second-Kind Fredholm Integral Equations Using a Combination of Reinforcement Learning and Context-Free Grammars  
**License:** MIT

---

## Overview

This repository presents a **grammar-guided Symbolic Reinforcement Learning framework** using **Deep Q-Networks (DQN)** to automatically discover **closed-form symbolic solutions** for Fredholm integral equations of the second kind:

$$u(x) = f(x) + \lambda \int_0^1 K(x,t)\, u(t)\, dt, \quad x \in [0,1]$$

Unlike PINN or KAN-based methods that produce numerical black-box approximations, this framework **searches a symbolic expression space** using RL and returns an **interpretable, closed-form expression** for $u(x)$.

---

## Key Contributions

- A **Context-Free Grammar (CFG)** that defines the space of valid mathematical expressions using operators `{+, −, ×, ÷, sin, cos, exp, log}` and learnable coefficients — based on the Chomsky hierarchy (Type-2 grammars).
- A **DQN agent** that navigates symbolic tree construction via sequential actions, modeled as a Markov Decision Process.
- A **physics-informed residual loss** evaluated via Gauss–Legendre quadrature as the reward signal.
- **LBFGS** coefficient optimization embedded inside the fitness evaluation loop.
- Benchmarks against **PINN**, **RBF-PIKAN**, and **Legendre-PIKAN** baselines across 5 integral equation examples.

---

## Framework Architecture

```
Action Sequence (DQN)
        │
        ▼
CFG Expression Generator  →  Symbolic Expression (e.g., sin(w_0 * x) + w_1)
        │
        ▼
CFGBasedModel (PyTorch)   →  Differentiable computation graph
        │
        ▼
LBFGS Coefficient Solver  →  Fits w_0, w_1, ... to minimize residual
        │
        ▼
Gauss–Legendre Quadrature →  u(x) − f(x) − λ ∫K(x,t)u(t)dt
        │
        ▼
Reward = 1 / (MAE + 1e-5) →  Back to DQN for policy update
```

---

## Repository Structure

```
Symbolic-RL-Fredholm-Solver/
│
├── Example 1/                  # Baseline: linear kernel, exact solution u(x)=x
│   ├── Fredholm_SecKind_RL_DQN.py
│   ├── Plot.py
│   ├── RL_DQN_Results.txt
│   ├── RL_DQN_Results_smoothed.pdf
│   └── Readme.md
│
├── Example 2/                  # Oscillatory kernel: K(x,t) = sin(x−t)
│   └── ...
│
├── Example 3/                  # Smooth non-homogeneous: K(x,t) = exp(−(x−t)²)(1+0.5t²)
│   └── ...
│
├── Example 4/                  # Stability test: Gaussian kernel with λ=1.2 (>1)
│   └── ...
│
├── Example 5/                  # Physical application: distributed RC transmission line
│   └── ...
│
├── Other methods/              # Baseline comparisons
│   ├── PINN/                   # Physics-Informed Neural Network
│   └── PI-KAN/
│       ├── RBF_Gaussian/       # RBF kernel activation network
│       └── Legendre/           # Legendre polynomial activation network
│
├── LICENSE
└── README.md
```

---

## Examples Summary

| # | Kernel K(x,t) | f(x) | λ | Exact? | Episodes | Convergence Episode |
|---|---------------|------|---|--------|----------|-------------------|
| 1 | $e^t$ | $xe$ | 1.0 | $u(x)=x$ | 1,500 | — |
| 2 | $\sin(x-t)$ | $\sin(\pi x)$ | 0.8 | ✗ | 15,000 | 11,118 |
| 3 | $e^{-(x-t)^2}(1+0.5t^2)$ | $\sin(\pi x)$ | 0.7 | ✗ | 60,000 | 35,029 |
| 4 | $e^{-(x-t)^2}$ | $\sin(\pi x)$ | 1.2 | ✗ | 60,000 | 13,452 |
| 5 | $e^{-5\|x-t\|}$ (RC line) | $\sin(\pi x)$ | 0.6 | ✗ | 60,000 | — |



---

## Methodology

### 1. Symbolic Expression Generation (CFG)
A sequence of integer actions drives a context-free grammar to expand symbolic trees. Terminals include the variable `x` and learnable scalar coefficients `w_0, w_1, ...`. Five production rule choices:

| Action | Expansion |
|--------|-----------|
| 0 | `expr op expr` |
| 1 | `(expr op expr)` |
| 2 | `pre_op(expr)` |
| 3 | constant `w_i` |
| 4 | variable `x` |

Binary operators: `{+, −, ×, ÷}`. Unary functions: `{sin, cos, exp, log}`.

### 2. Deep Q-Network
- **State:** Fixed-length integer sequence (length 30) of past actions.
- **Action space:** `{0, 1, 2, 3, 4}`.
- **Network:** 3-layer MLP (128 → 128 → 5 neurons), trained with Adam.
- **Policy:** ε-greedy with ε₀=0.999, decay=0.9995, ε_min=0.995.
- **Replay buffer:** Capacity 10,000; batch size 500.
- **Target network:** Updated every 15 episodes.

### 3. Physics-Informed Fitness
For each candidate expression, a PyTorch `CFGBasedModel` is built and coefficients are optimized with **LBFGS** to minimize:

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^N \left(u(x_i) - f(x_i) - \lambda \sum_{j=1}^N w_j\, K(x_i, t_j)\, u(t_j)\right)^2$$

where $(t_j, w_j)$ are Gauss–Legendre quadrature nodes and weights.

### 4. Reward Signal
$$R = \frac{1}{\text{MAE} + 10^{-5}}$$

Convergence is declared when MAE < 0.01. The reward plot shows rare high-reward spikes on a log scale — these correspond to episodes where the RL agent discovers a well-fitting symbolic structure.

---

## Numerical Results

| Example | Residual (‖·‖₁) | Discovered Symbolic Form |
|---------|----------------|--------------------------|
| 1 | MAE < 0.01 | $u(x) \approx x$ (exact recovery) |
| 2 | 2.698 × 10⁻³ | $u(x) = c_0 + \sin(c_1 x) + \ldots$ |
| 3 | 4.732 × 10⁻³ | $u(x) = c_0(c_1(x - \cos(\exp(x)))) - \ldots$ |
| 4 | 4.975 × 10⁻³ | $u(x) = c_0 - \exp(\sin(c_1 + x + c_2)) - \ldots$ |
| 5 | — | RC-line voltage profile |

---

## Baseline Methods (`Other methods/`)

| Method | Architecture | Activation | Key property |
|--------|-------------|------------|-------------|
| PINN | MLP + tanh | Fixed | Purely numerical output |
| RBF-PIKAN | KAN layers | Learnable Gaussian RBF | Better local adaptability |
| Legendre-PIKAN | KAN layers | Legendre polynomials | Spectral accuracy, faster convergence |

All baselines evaluated on the same 5 examples using MAE and RMS residual. The proposed symbolic method achieves **competitive accuracy** while additionally providing **interpretable closed-form expressions**.

---

## DQN Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate (Adam) | 0.01 |
| Discount factor γ | 0.9 |
| Initial ε | 0.999 |
| ε decay | 0.9995 |
| ε minimum | 0.995 |
| Batch size | 500 |
| Replay buffer | 10,000 |
| Target update freq | 15 episodes |
| State size | 30 |
| Action space size | 5 |
| Hidden layers | 2 × 128 neurons |
| Invalid expression penalty | 100 |
| Failed solve penalty | 10 |

---

## Getting Started

### Requirements

```bash
pip install torch numpy scipy
```

### Run an Example

```bash
cd "Example 1"
python Fredholm_SecKind_RL_DQN.py
```

Results are logged to `RL_DQN_Results.txt`. Each convergence event records the episode number, action sequence, discovered symbolic expression, and MAE.

To visualize the reward convergence curve:

```bash
python Plot.py
```

> ⚠️ **Runtime Warning:** Examples 2–5 require several hours of CPU time (2–10 hours). This is expected behavior for symbolic RL search — the agent must evaluate thousands of randomly generated symbolic expressions before converging. Consider reducing the `episodes` parameter for quick testing.

---

## Limitations

- **Computational cost:** RL-based symbolic search is inherently expensive; each episode requires LBFGS optimization of a newly constructed PyTorch graph.
- **Grammar expressivity:** The current CFG is fixed; richer grammars increase the search space exponentially.
- **No GPU acceleration:** The current implementation runs on CPU. Parallelizing expression evaluation across GPUs would significantly reduce runtimes.
- **Convergence not guaranteed:** Like all RL methods, convergence is stochastic and depends on hyperparameters and random seeds.

---

## Citation

If you use this code in your research, please cite:

```
Khalilpour, S. (2025). Solving Second-Kind Fredholm Integral Equations Using
a Combination of Reinforcement Learning and Context-Free Grammars.
M.Sc. Thesis, Shahid Beheshti University, Department of Computer Science.
Supervisor: Dr. Kourosh Parand.
```

---

## Related Work

- Physics-Informed Neural Networks (PINNs): Raissi et al., 2019.
- Kolmogorov–Arnold Networks (KAN): Liu et al., 2024.
- FIE-500k symbolic dataset for Fredholm equations: Scientific Data, 2025.
