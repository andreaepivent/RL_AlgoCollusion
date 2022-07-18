import numpy as np 
from IPython.display import clear_output
import time
from profitquantity import quantity_compute, profit_compute
from copy import deepcopy
import random
from find_state import find_rowindex

def q_learning_2agents(alpha, beta, criterion, criterion_final, n_episodes, S, A, q_table, ci, ai, mu, a0, delta):
    """Training 2 agents
    Arguments:
        alpha: learning rate
        beta: experimentation parameter
        criterion: stopping criterion - number of iterations without price change
        criterion_final: stopping criterion - stops episode after this number of iterations in any case
        n_episodes: number of episodes
        S: state space
        A: action space 
        q_table: initial q-matrix
        
    Returns:
        q_info: array containing prices and profits at every iteration for every episode
        q_tables1: final q_matrix for agent 1
        q_tables2: final q_matrix for agent 2"""
    
    # Compute execution time
    start_time = time.time()
    
    # Retrieve state and action space size
    state_space = len(S)
    action_space = len(A)
    
    # Store info in array - states, prices for both agents
    q_info = np.zeros((criterion_final,4*n_episodes))

    # Initial Q_tables
    q_tables1 = np.zeros([state_space, action_space])
    q_tables2 = np.zeros([state_space, action_space])
    
    for j in range(n_episodes):

        # Initialize Q-tables for both agents
        q_table_a1 = deepcopy(q_table)
        q_table_a2 = deepcopy(q_table)

        # Set counter for stop criterion
        count_a1 = 0
        count_a2 = 0

        # The initial state is picked randomly
        state = random.randint(0, state_space-1)

        # Store initial state in dataframe
        q_info[0,j*4] = S[state][0]
        q_info[0,(j*4)+1] = S[state][1]

        # Initialize matrix for keeping track of argmax_p q
        stab1 = np.full([state_space],-1)
        stab2 = np.full([state_space],-1)

        # Initialize convergence
        convergence = False

        # Start iteration
        i = 1    

        # While we didn't reach convergence
        while convergence == False:

            # Time-declining exploration rate
            epsilon = np.exp(-beta*i) # greedy parameter 

            ## Experimentation-exploitation trade-off
            # trade-off for agent 1
            if random.uniform(0, 1) < epsilon:
                action_a1 = random.randint(0, action_space-1) # Explore action space
            else:
                action_a1 = np.argmax(q_table_a1[state]) # Exploit learned values

            # trade-off for agent 2
            if random.uniform(0, 1) < epsilon:
                action_a2 = random.randint(0, action_space-1) # Explore action space
            else:
                action_a2 = np.argmax(q_table_a2[state]) # Exploit learned values

            # After actions are taken, retrieve new prices and next state
            p1, p2 = A[action_a1], A[action_a2]
            next_state = find_rowindex(S,p1,p2) # We find the row index associated with these two new prices

            # Retrieve rewards (= profits)
            reward_a1 = profit_compute(p1,p2,ci,ai,mu,a0)
            reward_a2 = profit_compute(p2,p1,ci,ai,mu,a0)

            # Store in array - Begin at i = 1
            q_info[i,j*4] = p1
            q_info[i,(j*4)+1] = p2
            q_info[i,(j*4)+2] = reward_a1
            q_info[i,(j*4)+3] = reward_a2

            # Check convergence - If 
            a1_argmax = np.argmax(q_table_a1[state]) 
            a2_argmax = np.argmax(q_table_a2[state]) 

            if a1_argmax == stab1[state]:
                count_a1 += 1
            else:
                count_a1 = 0
                stab1[state] = a1_argmax # reinitialization

            if a2_argmax == stab2[state]:
                count_a2 += 1
            else:
                count_a2 = 0
                stab2[state] = a2_argmax

            if (count_a1 >= criterion) & (count_a2 >= criterion):
                convergence = True

            # Updating Q-tables
            # Agent 1
            old_value_a1 = q_table_a1[state, action_a1]
            next_max_a1 = np.max(q_table_a1[next_state])

            new_value_a1 = (1 - alpha) * old_value_a1 + alpha * (reward_a1 + delta * next_max_a1)
            q_table_a1[state, action_a1] = new_value_a1

            # Agent 2
            old_value_a2 = q_table_a2[state, action_a2]
            next_max_a2 = np.max(q_table_a2[next_state])

            new_value_a2 = (1 - alpha) * old_value_a2 + alpha * (reward_a2 + delta * next_max_a2)
            q_table_a2[state, action_a2] = new_value_a2

            # Go to next step
            state = next_state

            # Display number of episodes and status of convergence
            if i % 100 == 0:
                clear_output(wait=True)
                print(f"Iteration: {i}")
                print(f"Episode: {j}")

            if (i < criterion_final) & (convergence == True):
                print("Process has converged")

            # If we didn't convergence after final threshold, we end the loop anyway
            if (i == criterion_final-1) & (convergence == False):
                print("Process has not converged") 
                convergence = True

            i += 1

        # Save every Q-table
        q_tables1 = np.concatenate((q_tables1,q_table_a1))
        q_tables2 = np.concatenate((q_tables2,q_table_a2))

        print(f"Training finished, episode: {j}")
    
    seconds = (time.time() - start_time)
    print("--- %s minutes ---" % (seconds/60))
    return([q_info,q_tables1,q_tables2])