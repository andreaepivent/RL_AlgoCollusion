import numpy as np 
import profitquantity

def init_Q(state_space,action_space,A,ci,ai,mu,a0,delta,n):
	q_table = np.zeros([state_space, action_space])
	b = 0 # loop over column
	sum_profit = 0
	for l in range(state_space):
	    for i in A:
	        for j in A:
	            profit = profitquantity.profit_compute(i,j,ci,ai,mu,a0)
	            sum_profit += profit
	        denom = (1-delta)*(action_space**(n-1))
	        q_table[l,b] = sum_profit/denom
	        sum_profit = 0
	        b += 1
	    b = 0
	return(q_table)