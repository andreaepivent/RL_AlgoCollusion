import numpy as np 

def quantity_compute(action_agent,action_opponent,ai,mu,a0):
    num = np.exp((ai-action_agent)/mu)
    denom = np.exp((ai-action_agent)/mu) + np.exp((ai-action_opponent)/mu) + np.exp(a0/mu)
    return(num/denom)
    
def profit_compute(action_agent,action_opponent,ci,ai,mu,a0):
    return((action_agent-ci)*quantity_compute(action_agent,action_opponent,ai,mu,a0))

def extra_profit_compute(p1,p2,ci,ai,mu,a0,profit_M,profit_N):
    """Compute extra-profit gain compared to Bertrand-Nash profit
    
    Arguments:
        p1: price of agent 1
        p2: price of agent 2
        ci: cost
        ai: demand parameter
        mu: horizontal diff
        a0: demand parameter - outside option
        profit_M: monopoly profit
        profit_N: B-N profit"""
    
    return((((profit_compute(p1,p2,ci,ai,mu,a0) + profit_compute(p2,p1,ci,ai,mu,a0))/2)-profit_N)/(profit_M-profit_N))