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

    Sampeling = 20
    Epochs = 50
    a,b=0,1

    def CheckModel():
        x_test = torch.linspace(a, b, 10).reshape(-1, 1)
        predict = (model.forward(x_test)).detach()
        if(predict.shape == torch.Size([1])):
            return False
        return True
    
    def Exact(x):
        return x
    
    def K(y, x):
        return torch.exp(y)

    def Fredholm_integral_eq(y, x):
        g = x * torch.e**1
        roots, weights = roots_legendre(Sampeling)
        weights = torch.tensor(weights, dtype=torch.float)
        int_term = (b - a) / 2 * K(y, x).flatten().dot(weights)
        return y + (x * int_term - g)

    if(CheckModel()==False):
       return torch.tensor(Invalid_Expr_Penalty).float()
    
    x = torch.linspace(a, b, Sampeling, requires_grad=True).reshape(-1, 1)

    optimizer = optim.LBFGS(list(model.parameters()), lr=0.1)

    def Closure():
        y = model.forward(x)
        # Compute the integral terms using trapezoidal rule
        integral_term = Fredholm_integral_eq(y, x)
        # Define the residual based on the integral equation
        residual = integral_term
        #initial = y[0] - 0
        loss = (residual**2).mean() #+ initial**2

        optimizer.zero_grad()
        loss.backward()
        return loss
    
    def Validation():
        x_test = torch.linspace(a,b, 33).reshape(-1, 1)
        exact = Exact(x_test)
        predict = model.forward(x_test).detach().numpy()
        error = exact - predict

        MAE = torch.abs(error).mean()
        return MAE
     
    for i in range(Epochs):
        optimizer.step(Closure)
        
    return torch.tensor(Validation()).float()

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
                    f.write(f"Expression: {generate_expression(action_sequence.copy())}\nSolution: {solution}\nMAE: {mae:.8f} | Reward: {reward:.2f}\n")
                    f.write("-"*40 + "\n")
                break
        
        # Update target network periodically
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode + 1}/{episodes}: Total Reward = {total_reward}")
        total_rewards.append(total_reward)
    with open(Log_Directory, "a") as f:
        f.write(f"Time: {datetime.datetime.now()} Total rewards: {total_rewards} \n")

# Run DQN Training
episodes = 1500
max_steps = 30
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
