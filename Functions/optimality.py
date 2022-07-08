import numpy as np
from find_state import find_rowindex
from init_Q import init_Q
from profitquantity import quantity_compute, profit_compute
from prices import get_last_price, get_forward_price

def optimality(agent,q_table,q_table_opp,init_p,init_p_opp,n_iterations,q_info,state_space,action_space,A,S,ci,ai,mu,a0,delta,alpha,n,n_episodes):
    """Compute optimality measure as defined in Klein (2021)
    Idea: fix rival's strategy to its limit strategy and compute theoretical Q-matrix for the non-fixed agent."""
    
    optim = []

    for j in range(n_episodes): 
    
        q = q_table[(j+1)*225:(j+1)*225+225,:]
        q_opp = q_table_opp[(j+1)*225:(j+1)*225+225,:]
    
        # Initialize Q-matrix of agent
        q_table = init_Q(state_space,action_space,A,ci,ai,mu,a0,delta,n)

        # Find last state and optimal action response according to limit strategy
        s_optim = find_rowindex(S,init_p[j][0],init_p_opp[j][0]) 
        a_optim = np.argmax(q[s_optim])
      
        # Loop over every state/action until convergence
        for act in A:

            # Initialize convergence criteria
            count_convergence = 0
            convergence = False

            while convergence == False:

                p,p_opp = act, A[np.argmax(q_opp[s_optim])] # Q-matrix of opponent doesn't change, play according to limit strategy
                next_state = find_rowindex(S,p,p_opp) # We find the row index associated with these two new prices
                action = np.where(A == p)[0][0] # get index associated to p1

                # Rewards
                reward = profit_compute(p,p_opp,ci,ai,mu,a0)

                # Updating Q-table - for non fixed agent only
                old_value = q_table[s_optim, action]
                next_max = np.max(q_table[next_state])

                new_value = (1 - alpha) * old_value + alpha * (reward + delta * next_max)
                q_table[s_optim, action] = new_value

                # We always stick to the same state
                #state = next_state

                diff = abs(old_value-new_value)

                if diff < 1e-4:
                    count_convergence += 1
                else:
                    count_convergence = 0

                if count_convergence == 100: # doesn't change for at least 100 iterations
                    convergence = True
                
        optim.append(q[s_optim,a_optim]/np.max(q_table[s_optim]))
        
    return(optim) # compare limit strategy and theoretical Q-matrix