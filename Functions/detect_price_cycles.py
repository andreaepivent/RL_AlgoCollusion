import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

def is_k_periodic(arr, k):
    """Check whether an array is k-periodic
    
    Arguments:
        arr: a numpy array or a list
        k: cycle length
        
    Returns:
        True/False"""
    
    if len(arr) < k // 2:  # we want the returned part to repeat at least twice... otherwise every list is periodic (1 period of its full self)
        return False

    return all(x == y for x, y in zip(arr, cycle(arr[:k])))

def price_cycle(prices,n_episodes):
    """Find cycle length for every episode
    
    Arguments:
        prices: contain n forward prices for each episode
        n_episodes: number of episodes
    
    Returns:
        cycle: cycle length for each episode"""
    
    cycle = np.zeros((n_episodes,))
    for j in range(n_episodes):
        cycle_found = False
        k = 1
        while cycle_found == False:
            cycle_found = is_k_periodic(prices[j],k)
            k += 1
            
        cycle[j] = k-1
        
    return(cycle)

def graph_cycle(price1,price2,p_N,p_M):
    """Graph forward prices of both agents"""
    
    graph = plt.plot(price1,marker="o")
    graph2 = plt.plot(price2,marker="o",color="green")

    plt.axhline(y=p_N,alpha=0.4,ls="--",color="black",label="Static Nash")
    plt.axhline(y=p_M,alpha=0.4,ls="--",color="red",label="Cartel")

    plt.xlabel('Period')
    plt.ylabel('Price')

    plt.legend(loc='lower right')
    plt.show()