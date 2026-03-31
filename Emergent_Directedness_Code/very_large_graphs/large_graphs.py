import numpy as np
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
import pandas as pd
import seaborn as sns
import random
from itertools import product
import networkx as nx
from joblib import dump
import gc
import sys
import os

#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))
import CPC_package as CPC

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import re
import multiprocessing
from networkx.generators.community import LFR_benchmark_graph

max_cores = multiprocessing.cpu_count()
print(f"Maximum number of cores available: {max_cores}")

CORES = 32

T_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 1, 2, 3, 4, 5]
sweep_value = 0.5
seed_function = CPC.randomFactorSeed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

CPL_configs_social = [
    # (n,      m, p,   random_portion)
    (5000,     5, 0.4, 20/5000),
    (10000,    5, 0.4, 20/10000),
    (20000,    5, 0.4, 20/20000)
]

CPL_graphs = [(nx.powerlaw_cluster_graph(n, m, p, seed=SEED), rp) for n, m, p, rp in CPL_configs_social]

results = []
tqdm_bar = tqdm(total=(len(CPL_graphs)*len(T_values)))
for graph, random_portion in CPL_graphs:
    #trigger the garbage collector to avoid memory issues
    gc.collect()
    name = 'CPL'
    for T in T_values:
        tqdm_bar.update(1)
        handler = CPC.CpcHandler(graph, cores=CORES, seed_function=seed_function, sweeps=sweep_value, random_seed_np=SEED)
        handler.random_portion = random_portion
        handler.to_dict_representation()
        handler.setThresholds(T)
        handler.setRandomFactor(0)
        handler.calcCPC()
        #result = handler.getNetworkWithCPC()
        sym = handler.calc_symmetry()

        #calculate the spreading density
        spreading_density = handler.getSpreadingDensity()
        print(spreading_density)

        #only append if spreading density is above a certain threshold
        results.append((name, len(graph.nodes), len(graph.edges), T, sym, spreading_density))

        df = pd.DataFrame(results, columns=['name', 'number_of_nodes', 'number_of_edges', 'T', 'Symmetry', 'Spreading_Density'])
        dump(df, 'large_CPL.joblib')
        print('finished')
