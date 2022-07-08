import numpy as np 

def quantity_compute(action_agent,action_opponent,ai,mu,a0):
    num = np.exp((ai-action_agent)/mu)
    denom = np.exp((ai-action_agent)/mu) + np.exp((ai-action_opponent)/mu) + np.exp(a0/mu)
    return(num/denom)
    
def profit_compute(action_agent,action_opponent,ci,ai,mu,a0):
    return((action_agent-ci)*quantity_compute(action_agent,action_opponent,ai,mu,a0))