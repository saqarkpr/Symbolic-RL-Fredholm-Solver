# 🧩 Other Methods

This folder contains the **baseline numerical models** used to compare with the proposed **Symbolic DQN Solver** for Fredholm integral equations of the second kind.

The goal of these methods is to evaluate the accuracy and convergence behavior of the proposed symbolic approach against standard physics-informed neural architectures.

---

## 📘 Included Models

### 1️⃣ PINN (Physics-Informed Neural Network)
The standard PINN architecture is used as the primary numerical reference.
It consists of a multi-layer perceptron (MLP) with `tanh` activation functions and is trained by minimizing the physics-based residual of the integral equation.
PINN provides a purely numerical solution \( $u(x)$ \) without an explicit symbolic form.

---

### 2️⃣ RBF-PIKAN (Physics-Informed Kernel Activation Network with RBF Basis)
This version extends the PINN by replacing the fixed activation function with a **learnable combination of radial basis functions (RBFs)**.
The RBF basis allows better local adaptability and smoother approximations in nonlinear regions of the integral kernel.

---

### 3️⃣ Legendre-PIKAN (Physics-Informed Kernel Activation Network with Legendre Basis)
This model replaces the RBFs with **Legendre polynomial bases**, providing a more stable and spectrally accurate representation.
Thanks to the orthogonality of Legendre polynomials, this version achieves faster convergence and improved numerical stability for smooth solutions.

---

## 🧠 Purpose of Comparison
All three models are used to benchmark:
- Mean Absolute Error (MAE)
- Convergence rate
- Computational efficiency  
against the **Symbolic DQN framework**, which generates interpretable closed-form (symbolic) solutions.

---

**Directory structure:**
```

Other methods/
├── PINN/             # Classical physics-informed neural network
└── PI-KAN/
    ├── RBF_Gaussian/          # RBF-based PIKAN implementation
    └── Legendre/     # Legendre-based PIKAN implementation

```

---

**Repository:** [Fredholm_Second_Kind_Symbolic_DQN](https://github.com/mmovahed/Fredholm_Second_Kind_Symbolic_DQN)
