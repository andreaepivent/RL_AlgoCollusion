import numpy as np
from find_state import find_rowindex

def get_last_price(n,q_info,n_iterations):
    """Retrieve last n prices for both agents
    
    Arguments:
        n: number of prices we wish to retrieve
        q_info: array containing prices and rewards for all iterations per episode
        
    Return:
        price1: last prices for agent 1
        price2: last prices for agent 2"""
  
    # Initialization
    episodes = int(q_info.shape[1]/4)
    price1 = np.zeros((episodes,n))
    price2 = np.zeros((episodes,n))
    
    for j in range(episodes):
        price1[j,:] = q_info[int(n_iterations[j])-n:int(n_iterations[j]),j*4]
        price2[j,:] = q_info[int(n_iterations[j])-n:int(n_iterations[j]),j*4+1]
        
    return([price1,price2])

def get_forward_price(n,q_table_1,q_table_2,q_info,n_iterations,S,A): 
    """Get forward n prices for both agents
    
    Arguments:
        n: number of prices we wish to retrieve
        q_table_1: final q-matrix of agent 1
        q_table_2: final q-matrix of agent 2
        q_info: array containing prices and rewards for all iterations per episode
        n_iterations: array containing number of iterations made in each episode
        S: state array
        A: action array
        
    Return:
        price1: forward prices of agent 1
        price2: forward prices of agent 2"""
    
    # Initialization
    episodes = int(q_info.shape[1]/4)
    price1 = np.zeros((episodes,n))
    price2 = np.zeros((episodes,n))
    
    for j in range(episodes):
        # Retrieve q-tables for specific episode
        q1 = q_table_1[(j+1)*225:(j+1)*225+225,:]
        q2 = q_table_2[(j+1)*225:(j+1)*225+225,:]
    
        # Retrieve last state
        p1 = get_last_price(1,q_info,n_iterations)[0][j][0]
        p2 = get_last_price(1,q_info,n_iterations)[1][j][0]
        state = find_rowindex(S,p1,p2) 

        # N iterations where learning is stopped 
        for t in range(n):
            action_a1 = np.argmax(q1[state])
            action_a2 = np.argmax(q2[state])
            price1[j,t] = A[action_a1]
            price2[j,t] = A[action_a2]
            state = find_rowindex(S,A[action_a1],A[action_a2]) # new state

    return([price1,price2])