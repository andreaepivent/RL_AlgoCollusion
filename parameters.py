# Set working directory
path = os.getcwd()

# Import custom functions
from functions import *

# Economic parameters
n = 2 # number of agents/firms
m = 15 # number of possible prices
ci = 1 # cost
ai = 2 # demand parameter
a0 = 0 # demand parameter - outside option
mu = 1/4 # horizontal differenciation
delta = 0.95 # discount factor
ksi = 0.1 # interval range for prices
k = 1 # memory width 
p_N = nash_price()
p_M = coop_price()

state_space = m**(n*k) # cardinal of state space
action_space = m # cardinal of action space

# Q-learning parameters
# Hyperparameters
alpha = 0.15 # learning rate - baseline scenario
beta = 4*10**-6 # experimentation parameter - baseline scenario

# Stop criterion
criterion = 10**4

# Stop in any case
criterion_final = 15*10**5 

# Number of episodes
n_episodes = 100