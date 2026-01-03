import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from scipy.special import roots_legendre
import datetime
import re
from torch.autograd import Variable

seed = 42424242
# random.seed(seed)
# os.environ["PYTHONHASHSEED"] = str(seed)
# np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

Invalid_Expr_Penalty = 100
NotSolve_Penalty = 10
regularization = 1e-5
Log_Directory = "RL_DQN_Results.txt"
# Expression generation
i = 0
def generate_expression(a):
    def expr(a):
        if not a:
            return "<expr>"

        choice = a.pop(0)

        if choice == 0:
            # expr op expr
            return f"{expr(a)} {op(a)} {expr(a)}"
        elif choice == 1:
            # (expr op expr)
            return f"({expr(a)} {op(a)} {expr(a)})"
        elif choice == 2:
            # pre-op(expr)
            return f"{pre_op(a)}({expr(a)})"

        elif choice == 3:
            return const()
        else:
            # var
            return var()


    def op(a):
        if not a:
            return "<op>"
        choice = a.pop(0) % 4
        return ['+', '-', '*', '/'][choice]

    def pre_op(a):
        if not a:
            return "<pre_op>"
        choice = a.pop(0) % 4
        return ['sin', 'cos', 'exp', 'log'][choice]

    def var():
        u = 'x'        
        return u
    
    def const():
        global i        
        u = f"w_{i}"
        i = i + 1
        return u
    
    return expr(a.copy())

def HasOnlyTerminal(expr):
    return "<pre_op2>" not in expr and "<pre_op>" not in expr and "<op>" not in expr and "<expr>" not in expr

def HasVariable(expr):
    return "x" in expr

def HasCoeff(expr):
    return "w" in expr


def is_valid_expression(expr):
    return HasOnlyTerminal(expr) and HasVariable(expr) and HasCoeff(expr)


# Neural network model based on symbolic expression
class CFGBasedModel(nn.Module):
    def __init__(self, expression):
        super(CFGBasedModel, self).__init__()
        self.original_expression = expression
        self.weights = nn.ParameterDict()
        for wi in re.findall(r'w_\d+', expression):
            if wi not in self.weights:
                self.weights[wi] = nn.Parameter(torch.zeros(1))

        self.functions = {
            'sin': torch.sin,
            'cos': torch.cos,
            'exp': torch.exp,
            'log': torch.log,
        }

    def forward(self, x):
        return self._build_computation_graph(self.original_expression, x)

    def getResult(self):
        expr = self.original_expression
        for name, param in self.weights.items():
            expr = expr.replace(name, str(param.data[0].item()))
        return expr

    def _build_computation_graph(self, expression, x):
        for name, param in self.weights.items():
            expression = expression.replace(name, f'self.weights["{name}"]')
        expression = expression.replace('sin', 'self.functions["sin"]')
        expression = expression.replace('cos', 'self.functions["cos"]')
        expression = expression.replace('exp', 'self.functions["exp"]')
        expression = expression.replace('log', 'self.functions["log"]')
        expression = expression.replace('x', 'x')
        computation_graph = eval(expression)
        return computation_graph

# Physics-informed neural network (PINN) for Fredholm integral equation
def Solve(model):
    from scipy.special import roots_legendre

    # ---------------- config (مثل قبل) ----------------
    Sampeling = 20     # تعداد نقاط هم برای x و هم برای کوادراتور
    Epochs = 50
    a, b = 0.0, 1.0
    lambda_ = 0.8      # λ مثال دومت

    # ---------------- چک حداقلی مدل ----------------
    def CheckModel():
        x_test = torch.linspace(a, b, 10).reshape(-1, 1)
        predict = (model.forward(x_test)).detach()
        return predict.ndim == 2 and predict.shape[0] > 1

    if not CheckModel():
        return torch.tensor(float(Invalid_Expr_Penalty)).float()

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


# Hyperparameters
alpha = 0.01  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.999  # Exploration rate
epsilon_decay = 0.9995
epsilon_min = 0.995
batch_size = 500
memory_size = 10000
target_update_freq = 15  # How frequently to update the target network
max_state_size = 30  # Maximum length of the state sequence

# Action space (0, 1, 2, 3 , 4)
actions = [0, 1, 2, 3 , 4]

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Memory for Experience Replay
memory = deque(maxlen=memory_size)

# Save experiences to memory
def store_experience(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# Sample a batch from memory
def sample_batch():
    return random.sample(memory, batch_size)

# Q-Function Update
def optimize_model(policy_net, target_net, optimizer):
    if len(memory) < batch_size:
        return

    # Sample a batch of experiences
    batch = sample_batch()
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert to PyTorch tensors
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    # Calculate current Q values
    q_values = policy_net(states).gather(1, actions)

    # در DQN، انتخاب اکشن و محاسبه Q-value از Target Network انجام می‌شود
    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1) # محاسبه فقط براساس شبکه target که نشون میده DQN است.

    # محاسبه هدف Q-value
    target_q_values = rewards + (gamma * next_q_values * (1 - dones))

    # Loss and optimize
    loss = nn.MSELoss()(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Fitness function for evaluating an expression
def fitness(action_sequence):
    expr = generate_expression_from_actions(action_sequence)
    if not is_valid_expression(expr):
        return float(Invalid_Expr_Penalty), ''
    model = CFGBasedModel(expr)
    mae = Solve(model)
    if torch.isnan(mae):
        return float(NotSolve_Penalty), ''
    return mae, model.getResult()

# Padding function to ensure state has fixed size
def pad_state(state, max_size):
    if len(state) < max_size:
        return state + [0] * (max_size - len(state))
    else:
        return state[-max_size:]  # Keep only the last `max_size` elements
    
# Select action using epsilon-greedy policy
def select_action(state, policy_net):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)  # Exploration
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = policy_net(state_tensor)
            return q_values.max(1)[1].item()  # Exploitation

# Generate an expression from a sequence of actions
def generate_expression_from_actions(action_sequence):
    return generate_expression(action_sequence.copy())

# Main DQN Training Loop
def dqn_training(episodes, max_steps, state_size, action_size):
    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())  # Initialize target net with same weights
    target_net.eval()  # Set target network to evaluation mode
    optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
    total_rewards = []
    for episode in range(episodes):
        state = [0] * state_size  # Initialize state with zeros (or based on your state space)
        action_sequence = []
        total_reward = 0
        done = False

        for step in range(max_steps):
            # Pad the state to ensure fixed size
            padded_state = pad_state(state, max_state_size)

            # Select action using epsilon-greedy policy
            action = select_action(padded_state, policy_net)
            action_sequence.append(action)
            
            # Generate new state (expression) after action
            next_state = state.copy()
            next_state.append(action)

            # Evaluate the new state (expression) and get reward
            mae, solution = fitness(action_sequence)
            # reward = -mae * 100000
            reward = 1/(mae+regularization) 
            total_reward += reward  # Update the total reward for this episode

            done = mae < 1

            # Store experience in memory
            store_experience(padded_state, action, reward, pad_state(next_state, max_state_size), done)

            # Update state
            state = next_state

            # Optimize the model
            optimize_model(policy_net, target_net, optimizer)

            if mae < 0.01:
                with open(Log_Directory, "a") as f:
                    # f.write(f"Time: {datetime.datetime.now()} Expression: y(x)={solution} - MAE = {mae}\n")
                    f.write(f"Episode: {episode+1} | Step: {step+1} | Time: {datetime.datetime.now()}\n")
                    f.write(f"ActionSeq: {action_sequence}\n")
                    f.write(f"Expression: {generate_expression(action_sequence)}\nSolution: {solution}\nMAE: {mae:.8f} | Reward: {reward:.2f}\n")
                    f.write("-"*40 + "\n")
                break
            if(HasOnlyTerminal(generate_expression(action_sequence))):
                break; 
        # Update target network periodically
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode + 1}/{episodes}: Total Reward = {total_reward}")
        total_rewards.append(total_reward)
    with open(Log_Directory, "a") as f:
        f.write(f"Time: {datetime.datetime.now()} Total rewards: {total_rewards} \n")

# Run DQN Training
episodes = 15000
max_steps = 100
state_size = max_state_size  # Use the fixed size for state
action_size = len(actions)
now = datetime.datetime.now()
with open(Log_Directory, "a") as f:
    a = "-"*80 + "\n"
    f.write(a+f"Start time: {now}\n")
dqn_training(episodes, max_steps, state_size, action_size)
now = datetime.datetime.now()
with open(Log_Directory, "a") as f:
    a = "-"*80 + "\n"

    f.write(f"End time: {now}\n"+a)
