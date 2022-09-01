# Set working directory
import os
path = os.getcwd()

# Import packages
exec(open(path+"/packages.py").read())

# Import parameters
exec(open(path+"/parameters.py").read())

# Compute Nash price
def fixed_point(p):
    return(ci + mu/(1 - (n + np.exp((a0 - ai + p) / mu))**(-1)))

def p_N():
    #return(round(optimize.fixed_point(fixed_point, [0])[0],2))
    return(optimize.fixed_point(fixed_point, [0])[0])

# Compute cooperation price
def industry_profit(p):
    """Returns negative industry profit"""
    return(-((p - ci) * np.exp((ai - p) / mu) / (n * np.exp((ai - p) / mu) + np.exp(a0 / mu))))

def p_M():
    res = optimize.minimize_scalar(industry_profit)
    #return(round(res.x,2))
    return(res.x)

# Compute quantities, profits and extra-profits
def quantity_compute(action_agent,action_opponent):
    num = np.exp((ai-action_agent)/mu)
    denom = np.exp((ai-action_agent)/mu) + np.exp((ai-action_opponent)/mu) + np.exp(a0/mu)
    return(num/denom)
    
def profit_compute(action_agent,action_opponent):
    return((action_agent-ci)*quantity_compute(action_agent,action_opponent))

def extra_profit_compute(p1,p2):
    """Compute extra-profit gain compared to Bertrand-Nash profit
    
    Arguments:
        p1: price of agent 1
        p2: price of agent 2"""
    
    profit_N = profit_compute(p_N(),p_N())
    profit_M = profit_compute(p_M(),p_M())
    return((((profit_compute(p1,p2) + profit_compute(p2,p1))/2)-profit_N)/(profit_M-profit_N))

# Initialize Q-matrix
def init_Q(A):
    q_table = np.zeros([state_space, action_space])
    b = 0 # loop over column
    sum_profit = 0
    for l in range(state_space):
        for i in A:
            for j in A:
                profit = profit_compute(i,j)
                sum_profit += profit
            denom = (1-delta)*(action_space**(n-1))
            q_table[l,b] = sum_profit/denom
            sum_profit = 0
            b += 1
        b = 0
    return(q_table)

# Find state
def find_rowindex(S,row,col):
    row_index = -1
    state_space = S.shape[0]
    for i in range(state_space):
            if (S[i,0] == row) & (S[i,1] == col):
                row_index = i
    return(row_index)

# Q-learning: 2 agents @jit()
def q_learning_2agents(S, A , q_table, beta=beta, alpha=alpha,n_episodes=n_episodes, criterion=criterion, criterion_final=criterion_final, save_info=True):
    """Training 2 agents
    Arguments:
        S: state space
        A: action space 
        q_table: initial q-matrix
        n_episodes: number of simulations to run
        criterion: stopping criterion (no changes in optimal actions)
        criterion_final: stops simulation in any case after this number of iterations
        save_info: Whether prices should be saved throughout iterations, default=True
        
    Returns:
        q_info: array containing prices and profits at every iteration for every episode
        conv_info: array containing number of iterations and last prices of both agents for each episode 
        q_tables1: final q_matrix for agent 1
        q_tables2: final q_matrix for agent 2"""
    
    # Compute execution time
    start_time = time.time()
    
    # Retrieve state and action space size
    state_space = len(S)
    action_space = len(A)
    
    if save_info == True:
        # Store info in array - states, prices for both agents
        q_info = np.zeros((criterion_final,2*n_episodes))

    # Store number of iterations
    conv_info = np.zeros((3,n_episodes))
      
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

        if save_info == True:
            # Store initial state in dataframe
            q_info[0,j*2] = S[state][0]
            q_info[0,(j*2)+1] = S[state][1]

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
            epsilon = math.exp(-beta*i) # greedy parameter 

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
            reward_a1 = profit_compute(p1,p2)
            reward_a2 = profit_compute(p2,p1)
            
            if save_info == True:
                # Store in array - Begin at i = 1
                q_info[i,j*4] = p1
                q_info[i,(j*4)+1] = p2

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
                conv_info[0,j] = i
                conv_info[1,j] = p1
                conv_info[2,j] = p2

            # If we didn't convergence after final threshold, we end the loop anyway
            if (i == criterion_final-1) & (convergence == False):
                #print("Process has not converged") 
                convergence = True
                conv_info[0,j] = i
                conv_info[1,j] = p1
                conv_info[2,j] = p2

            i += 1

        # Save every Q-table
        q_tables1 = np.concatenate((q_tables1,q_table_a1))
        q_tables2 = np.concatenate((q_tables2,q_table_a2))

        print(f"Training finished, episode: {j}")
    
    seconds = (time.time() - start_time)
    print("--- %s minutes ---" % (seconds/60))
    if save_info == True:
        return([q_info,conv_info,q_tables1,q_tables2])
    else:
        return([conv_info,q_tables1,q_tables2])

# Retrieve forward prices in a simulation
def get_forward_prices(x,q_table_1,q_table_2,conv_info,S,A,n_episodes=n_episodes): 
    """Get forward x prices for both agents
    
    Arguments:
        x: number of prices we wish to retrieve
        q_table_1: final q-matrix of agent 1
        q_table_2: final q-matrix of agent 2
        conv_info: array containing number of iterations and last prices of both agents for each episode 
        S: state array
        A: action array
        
    Return:
        price1: forward prices of agent 1
        price2: forward prices of agent 2"""
    
    # Initialization
    price1 = np.zeros((n_episodes,x))
    price2 = np.zeros((n_episodes,x))
    
    for j in range(n_episodes):
        # Retrieve q-tables for specific episode
        q1 = q_table_1[(j+1)*225:(j+1)*225+225,:]
        q2 = q_table_2[(j+1)*225:(j+1)*225+225,:]
    
        # Retrieve last state
        p1 = conv_info[1,j]
        p2 = conv_info[2,j]
        state = find_rowindex(S,p1,p2) 

        # N iterations where learning is stopped 
        for t in range(x):
            action_a1 = np.argmax(q1[state])
            action_a2 = np.argmax(q2[state])
            price1[j,t] = A[action_a1]
            price2[j,t] = A[action_a2]
            state = find_rowindex(S,A[action_a1],A[action_a2]) # new state

    return([price1,price2])

# Detect price cycles
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

def price_cycle(prices,n_episodes=n_episodes):
    """Find cycle length for every episode
    
    Arguments:
        prices: contain n forward prices for each episode
    
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

def graph_cycle(price1,price2):
    """Graph forward prices of both agents"""
    
    graph = plt.plot(price1,marker="o")
    graph2 = plt.plot(price2,marker="o",color="green")

    plt.axhline(y=p_N(),alpha=0.4,ls="--",color="black",label="Static Nash")
    plt.axhline(y=p_M(),alpha=0.4,ls="--",color="red",label="Cartel")

    plt.xlabel('Period')
    plt.ylabel('Price')

    plt.legend(loc='lower right')
    plt.show()
    
# Compute impulse function after price cut/increase
def impulse_function(cycles,scenario,deviation_rank,sample,q_table_1,q_table_2,conv_info,S,A,save_as,graph=True,n_seq=30,deviation_period=10,price_cap=True,n_episodes=n_episodes):
    
    """Generates impulse function graph after a deviation scenario (price cut or price raise) at period 10 from agent 1
    
    Arguments:
        cycles: array containing cycle length of each episode
        scenario: "cut" or "raise"
        deviation_rank: (cut or raise) deviation rank, must be positive
        sample: "full" (whole sample), "point" (restriction to episodes where prices have converged to a point)
        "point-asym" (point episodes with asymmetric strategies, "point-sym" (point episodes with symmetric strategies),
        "set" (restriction to episodes where prices have converged to a set
        q_table_1: final q-matrix of agent 1
        q_table_2: final q-matrix of agent 2
        conv_info: array containing number of iterations and last prices of both agents for each episode 
        S: state array
        A: action array
        save_as: name of the output graph
        graph: display graph or not, default is True
        n_seq: number of sequences
        deviation_period: should be larger than > 1 and lower than n_seq
        price_cap: whether we should include simulations already at or above monopoly price
    
    Returns:
        prices1, prices2: sequence of prices of agent 1 and 2"""

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
    
    loop_size = len(loop)
    non_appl = 0 # keep track of sessions already at or above monopoly price
    
    prices1 = np.zeros((loop_size,n_seq))
    prices2 = np.zeros((loop_size,n_seq))
    
    # start counter
    i = 0
    for j in loop:
        
        # We retrieve the forward prices
        f_price1, f_price2 = get_forward_prices(10,q_table_1,q_table_2,conv_info,S,A)
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

        # We do not consider simulations where agent 1 is already pricing at or above monopoly price
        if (f_price1[j][0] in [A[A.shape[0]-2],A[A.shape[0]-1]]) & (scenario == "raise") & (price_cap == True):
            non_appl += 1
        
        else:
            # Storing initial values
            prices1[i,0] = f_price1[j][0]
            prices2[i,0] = f_price2[j][0]

            for t in range(1,n_seq):

                if t == deviation_period: # we force a price cut from agent 1

                    index = np.where(A == p1)[0][0]

                    if scenario == "cut":
                        #p1 = A[max(int(index/deviation_ratio),0)]
                        p1 = A[max(int(index-deviation_rank),0)]
                    elif scenario == "raise":
                        #p1 = A[min(int(index*deviation_ratio),A.shape[0]-1)]
                        p1 = A[min(int(index+deviation_rank),A.shape[0]-1)]
                    action_a2 = np.argmax(q2[state])
                    p2 = A[action_a2]

                else:
                    action_a1 = np.argmax(q1[state])
                    action_a2 = np.argmax(q2[state])
                    p1, p2 = A[action_a1], A[action_a2]

                prices1[i,t] = p1
                prices2[i,t] = p2

                state = find_rowindex(S,p1,p2) # new state

            i += 1
            
    # Remove rows with zero - corresponds to simulations already at or above monopoly price
    prices1, prices2 = prices1[~np.all(prices1 == 0, axis=1)], prices2[~np.all(prices2 == 0, axis=1)]
    
    if graph == True:
        # Visualisation
        plt.plot(prices1.mean(axis=0), marker="o", label = "Deviating agent")
        plt.plot(prices2.mean(axis=0), marker="o", label = "Nondeviating agent")
        plt.legend(loc='lower right')
        #plt.title('Price '+scenario+' in period 10')
        plt.axvline(x=10,alpha=0.4,ls="--",color="black")

        plt.xlabel('Period')
        plt.ylabel('Price')
        plt.ylim(1.45, 1.95)
        plt.savefig(path+"/Output/"+save_as+".png", format="png")
        plt.show()
        
    print(non_appl)
    return(prices1,prices2)

# Compute impulse function after invitation to collude
def invitation_collude(cycles,q_table_1,q_table_2,conv_info,S,A,save_as,deviation_rank=1,graph=True,n_seq=30,deviation_period=10,x=6,release_agent=1,n_episodes=n_episodes):

    """Generates impulse function for following scenario:
    * t = deviation_period, agent 1 (exogenously) increases its price
    * t = deviation_period+1, agent 2 (exogenously) matches its price increase and agent 1 keeps (exogenously) its price level
    * t = deviation_period+x, one of the agent is released while the other keeps (exogenously) its price level
    
    Arguments:
        deviation_rank: (cut or raise) deviation rank, must be positive
        q_table_1: final q-matrix of agent 1
        q_table_2: final q-matrix of agent 2
        conv_info: array containing number of iterations and last prices of both agents for each episode 
        S: state array
        A: action array
        save_as: name of the output graph
        graph: display graph or not, default is True
        n_seq: number of sequences
        deviation_period: should be larger than > 1 and lower than n_seq
        x: duration before release (+1)
        release_agent: choose which agent is release
    
    Returns:
        prices1, prices2: sequence of prices of agent 1 and 2"""
    
    # We restrict to simulations that have converged to point for simplicity
    loop = np.where((cycles==1)|(cycles==1.5))[0]
    
    loop_size = len(loop)
    
    prices1 = np.zeros((loop_size,n_seq))
    prices2 = np.zeros((loop_size,n_seq))
    
    # start counter
    i = 0
    for j in loop:
        
        # We retrieve the forward prices
        f_price1, f_price2 = get_forward_prices(10,q_table_1,q_table_2,conv_info,S,A,n_episodes=n_episodes)
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

        # We do not consider simulations where agent 1 is already pricing at or above monopoly price
        if f_price1[j][0] in [A[A.shape[0]-2],A[A.shape[0]-1]]:
            pass
        
        else:
            # Storing initial values
            prices1[i,0] = f_price1[j][0]
            prices2[i,0] = f_price2[j][0]

            for t in range(1,n_seq):

                if t==deviation_period: # we force a pro-collusive deviation from agent 1

                    index = np.where(A == p1)[0][0]

                    p1 = A[min(int(index+deviation_rank),A.shape[0]-2)]

                    action_a2 = np.argmax(q2[state])
                    p2 = A[action_a2]

                elif (t > deviation_period) & (t<deviation_period+x): # we force agent 2 to match previous deviation (invitation accepted)
                    # agent 1 keeps its previous price
                    p2 = p1

                elif (t>=deviation_period+x): # we release agent 1 and check reaction, agent 2 keeps its previous price
                    if release_agent == 1:
                        action_a1 = np.argmax(q1[state])
                        p1 = A[action_a1]
                    elif release_agent == 2:
                        action_a2 = np.argmax(q2[state])
                        p2 = A[action_a2]

                else: # Initially, we let both agents behave greedily
                    action_a1 = np.argmax(q1[state])
                    action_a2 = np.argmax(q2[state])
                    p1, p2 = A[action_a1], A[action_a2]

                prices1[i,t] = p1
                prices2[i,t] = p2

                state = find_rowindex(S,p1,p2) # new state

            i += 1
    
    # Remove rows with zero - corresponds to simulations already at or above monopoly price
    prices1, prices2 = prices1[~np.all(prices1 == 0, axis=1)], prices2[~np.all(prices2 == 0, axis=1)]
    
    if graph == True:
        # Visualisation
        plt.plot(prices1.mean(axis=0), marker="o", label = "Deviating agent")
        plt.plot(prices2.mean(axis=0), marker="o", label = "Nondeviating agent")
        plt.legend(loc='lower left')
        plt.axvline(x=10,alpha=0.4,ls="--",color="black")

        plt.xlabel('Period')
        plt.ylabel('Price')
        plt.ylim(1.45, 1.9)
        plt.savefig(path+"/Output/"+save_as+".png", format="png")
        plt.show()

# Compute matrices of relative price changes after price cut/raise
def pricechanges_dict(i,scenario,cycles,q_table_1,q_table_2,conv_info,S,A):

    """Returns dictionnary with (predeviation price,deviation price) of deviating agent (agent 1 here) as a key and relative price change as a value for each episode and each deviation rank
    
    Arguments:
        i: agent
        scenario: "cut" or "raise"
        cycles: array containing cycle length of each episode
        q_table_1: q-matrix of agent 1
        q_table_2: q-matrix of agent 2
        conv_info: array containing number of iterations and last prices of both agents for each episode 
        S: state space
        A: action space
        """
    
    A_round = np.round(A,2)
    ep = [k for k in range(n_episodes)]

    # Create dictionnary
    pricechanges = dict()
    keys = list(product(A_round, repeat=2))
    for key in keys:
        pricechanges[key] = []

    for d in range(1,15):
        p = impulse_function(cycles,scenario,d,"full",q_table_1,q_table_2,conv_info,S,A,graph=False,n_seq=4,deviation_period=2,price_cap=False)
        p1 = np.round(p[0],2)
        p2 = np.round(p[1],2)
        for j in ep:
            if p1[j,1] == p1[j,2]:
                # If predeviation price == deviation price (can occur for min and max price)
                pass
            else:
                if i == 1:
                    # We compute relative price change
                    pricechanges[(p1[j,1],p1[j,2])].append(p1[j,3]/p1[j,2]-1) # postdeviation price/deviation price
                else:
                    # We compute relative price change and keep prices of agent 1 as key
                    pricechanges[(p1[j,1],p1[j,2])].append(p2[j,3]/p2[j,2]-1) # postdeviation price/deviation price
                    
                # If we reach lower or upper bound for deviation price, we switch to other episode
                if (p1[j,2] == A_round[0]) | (p1[j,2] == A_round[len(A)-1]):
                    ep.remove(j)
                    
            print((j,d))
    
    return(pricechanges)

def pricechanges_mat(pricechanges,A):
    """Returns matrix of relative price change after a deviation
    
    Arguments:
        pricechanges: a dictionnary containing (predeviation price,deviation price) as a key and post deviation price as a value
        A: action space"""
    
    mat = np.zeros((len(A),len(A)))
    A_round = np.round(A,2)
    
    for key in pricechanges:
        row = np.where(A_round==key[0])[0][0]
        column = np.where(A_round==key[1])[0][0]
        if np.isnan(np.mean(pricechanges[key])) == False:
            mat[row,column] = np.round(np.mean(pricechanges[key]),2) # average relative price change
            
    return(mat)

# Concatenate dictionnaries
def concatDict(dict1, dict2):
    ''' Concatenate dictionnaries that do not have value for same keys in list'''

    dict3 = {}
    for key, value in dict1.items():
        if dict1[key] == []:
            dict3[key] = dict2[key]
        if dict1[key] != []:
            dict3[key] = dict1[key]
    return dict3

# Robustness check for impulse functions
def impulse_function_robustness(cycles,scenario,deviation_rank,q_table_1,q_table_2,conv_info,S,A,graph=True,n_seq=30,deviation_period=10,price_cap=True):
    
    """Generates impulse function graph after a deviation scenario (price cut or price raise) at period 10 from agent 1
    
    Arguments:
        cycles: array containing cycle length of each episode
        scenario: "cut" or "raise"
        deviation_rank: (cut or raise) deviation rank, must be positive
        q_table_1: final q-matrix of agent 1
        q_table_2: final q-matrix of agent 2
        conv_info: array containing number of iterations and last prices of both agents for each episode 
        S: state array
        A: action array
        graph: display graph or not, default is True
        n_seq: number of sequences
        deviation_period: should be larger than > 1 and lower than n_seq
        price_cap: whether we should include simulations already at or above monopoly price"""
    
    for cycle in np.unique(cycles):
        loop = np.where(cycles==cycle)[0]
    
        loop_size = len(loop)
        non_appl = 0 # keep track of sessions already at or above monopoly price
    
        prices1 = np.zeros((loop_size,n_seq))
        prices2 = np.zeros((loop_size,n_seq))

        # start counter
        i = 0
        for j in loop:

            # We retrieve the forward prices
            f_price1, f_price2 = get_forward_prices(10,q_table_1,q_table_2,conv_info,S,A)
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

            # We do not consider simulations where agent 1 is already pricing at or above monopoly price
            if (f_price1[j][0] in [A[A.shape[0]-2],A[A.shape[0]-1]]) & (scenario == "raise") & (price_cap == True):
                non_appl += 1

            else:
                # Storing initial values
                prices1[i,0] = f_price1[j][0]
                prices2[i,0] = f_price2[j][0]

                for t in range(1,n_seq):

                    if t == deviation_period: # we force a price cut from agent 1

                        index = np.where(A == p1)[0][0]

                        if scenario == "cut":
                            #p1 = A[max(int(index/deviation_ratio),0)]
                            p1 = A[max(int(index-deviation_rank),0)]
                        elif scenario == "raise":
                            #p1 = A[min(int(index*deviation_ratio),A.shape[0]-1)]
                            p1 = A[min(int(index+deviation_rank),A.shape[0]-1)]
                        action_a2 = np.argmax(q2[state])
                        p2 = A[action_a2]

                    else:
                        action_a1 = np.argmax(q1[state])
                        action_a2 = np.argmax(q2[state])
                        p1, p2 = A[action_a1], A[action_a2]

                    prices1[i,t] = p1
                    prices2[i,t] = p2

                    state = find_rowindex(S,p1,p2) # new state

                i += 1

        # Remove rows with zero - corresponds to simulations already at or above monopoly price
        prices1, prices2 = prices1[~np.all(prices1 == 0, axis=1)], prices2[~np.all(prices2 == 0, axis=1)]

        if graph == True:
            # Visualisation
            plt.plot(prices1.mean(axis=0), marker="o", label = "Deviating agent")
            plt.plot(prices2.mean(axis=0), marker="o", label = "Nondeviating agent")
            plt.legend(loc='lower right')
            #plt.title('Price '+scenario+' in period 10')
            plt.axvline(x=10,alpha=0.4,ls="--",color="black")

            plt.xlabel('Period')
            plt.ylabel('Price')
            plt.ylim(1.45, 1.95)
            plt.show()

        print(non_appl)
        print(cycle)