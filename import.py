# Import relevant packages
import os
import time
import numpy as np
import pandas as pd
import random
from IPython.display import clear_output
from copy import deepcopy
import matplotlib.pyplot as plt
from itertools import cycle

# Import our custom functions
os.chdir(path)
from functions import find_rowindex, quantity_compute, profit_compute, extra_profit_compute, init_Q, q_learning_2agents, get_last_price, get_forward_price, is_k_periodic, price_cycle, graph_cycle, impulse_function