import numpy as np

# Define the grid size and actions
grid_size = 5
n_actions = 4  # Actions: up, down, left, right

# Initialize the Q-table with zeros
Q_table = np.zeros((grid_size * grid_size, n_actions))

alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor for future rewards
epsilon = 0.1  # Exploration rate for epsilon-greedy policy

# Reward matrix for the grid environment
rewards = np.full((grid_size * grid_size,), -1)  # -1 for every state
rewards[24] = 10  # Goal state
rewards[12] = -10  # Pitfall state

def epsilon_greedy_action(Q_table, state, epsilon):
  if np.random.uniform(0, 1) < epsilon:
    return np.random.randint(0, n_actions)  # Explore: random action
  else:
    return np.argmax(Q_table[state])  # Exploit: action with highest Q-value