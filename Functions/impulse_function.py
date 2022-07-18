import numpy as np
import matplotlib.pyplot as plt
from find_state import find_rowindex
from prices import get_last_price, get_forward_price

def impulse_function(cycles,n_episodes,scenario,deviation_ratio,sample,q_table_1,q_table_2,q_info,n_iterations,S,A):
    
    """Generates impulse function graph after a deviation scenario (price cut or price raise) at period 10 from agent 1
    
    Arguments:
        cycles: array containing cycle length of each episode
        n_episodes: number of episodes
        deviation_ratio: (cut or raise) deviation ratio, must be positive
        sample: "full" (whole sample), "point" (restriction to episodes where prices have converged to a point)
        "point-asym" (point episodes with asymmetric strategies, "point-sym" (point episodes with symmetric strategies),
        "set" (restriction to episodes where prices have converged to a set
    
    Returns:
        prices1, prices2: sequence of prices of agent 1 and 2"""
    
    n_seq = 30
    prices1 = np.zeros((n_episodes,n_seq))
    prices2 = np.zeros((n_episodes,n_seq))

    if sample == "full":
        loop = range(n_episodes)
        
    elif sample == "point":
        loop = np.where((cycles==1)|(cycles==1.5))[0]
    
    elif sample == "point-sym":
        loop = np.where(cycles==1)[0]
        
    elif sample == "point-asym":
        loop = np.where(cycles==1.5)[0]
        
    elif sample == "set":
        loop = np.where((cycles!=1)&(cycles!=1.5))[0]
        
    for j in loop:

        # We retrieve the forward prices
        f_price1, f_price2 = get_forward_price(n_seq,q_table_1,q_table_2,q_info,n_iterations,S,A)
        # Keep last one
        ## We do this because exploration can still occur towards end of episode, therefore it may take a few iterations for agents to converge to final strategies 
        ## It ensures that we do not observe weird patterns in the restricted case with sample converged to a point
        f_price1 = f_price1[:,f_price1.shape[1]-1:f_price1.shape[1]]
        f_price2 = f_price2[:,f_price2.shape[1]-1:f_price2.shape[1]]

        # We retrieve the appropriate Q-table
        q1 = q_table_1[(j+1)*225:(j+1)*225+225,:]
        q2 = q_table_2[(j+1)*225:(j+1)*225+225,:]

        # Find actual state
        state = find_rowindex(S,f_price1[j][0],f_price2[j][0])

        # Storing initial values
        prices1[j,0] = f_price1[j][0]
        prices2[j,0] = f_price2[j][0]

        for t in range(1,n_seq):

            if t == 10: # we force a price cut from agent 1

                index = np.where(A == f_price1[j][0])[0][0]

                if scenario == "cut":
                    p1 = A[max(int(index/deviation_ratio),0)]
                elif scenario == "raise":
                    p1 = A[min(int(index*deviation_ratio),A.shape[0]-1)]

                action_a2 = np.argmax(q2[state])
                p2 = A[action_a2]

            else:
                action_a1 = np.argmax(q1[state])
                action_a2 = np.argmax(q2[state])
                p1, p2 = A[action_a1], A[action_a2]

            prices1[j,t] = p1
            prices2[j,t] = p2

            state = find_rowindex(S,p1,p2) # new state

    # Visualisation
    plt.plot(prices1.mean(axis=0), marker="o", label = "Agent 1")
    plt.plot(prices2.mean(axis=0), marker="o", label = "Agent 2")
    plt.legend(loc='lower right')
    plt.title('Price cut in period 10 from agent 1')
    plt.axvline(x=10,alpha=0.4,ls="--",color="black")

    plt.xlabel('Period')
    plt.ylabel('Price')
    plt.show()
            
    return(prices1,prices2)