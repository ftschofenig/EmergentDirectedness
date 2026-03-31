import sys
import os
import random
import re
import multiprocessing
import warnings

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from itertools import product
from pathos.multiprocessing import ProcessingPool as Pool
from joblib import dump
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, ConstantInputWarning

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))
import CPC_package as CPC

warnings.filterwarnings("ignore", category=ConstantInputWarning)
np.seterr(invalid='ignore', divide='ignore')
random.seed(42)
np.random.seed(42)

max_cores = multiprocessing.cpu_count()
print(f"Maximum number of cores available: {max_cores}")

AddHealth_Graphs = CPC.getAddHealthGraphs()
Banerjee_graphs = CPC.getBanerjeeGraphs()

CORES = 128 #max_cores

T_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 1, 2, 3]

sweep_value = 1

seed_function = CPC.randomFactorSeed

results = []
node_log = []
edge_log = []

print('Starting calculations on AddHealth...')
tqdm_bar = tqdm(total=(len(AddHealth_Graphs)*len(T_values)))
for name, graph in AddHealth_Graphs.items():
    for T in T_values:
        tqdm_bar.update(1)
        handler = CPC.CpcHandler(graph, cores=CORES, seed_function=seed_function, sweeps=sweep_value, random_seed_np=42)
        handler.to_dict_representation()
        handler.setThresholds(T)
        handler.setRandomFactor(0)
        handler.calcCPC()
        result_1 = handler.getNetworkWithCPC()
        sym1 = handler.calc_symmetry()

        handler2 = CPC.CpcHandler(graph, cores=CORES, seed_function=seed_function, sweeps=sweep_value, random_seed_np=42)
        handler2.symmetric_update = False
        handler2.to_dict_representation()
        handler2.setThresholds(T)
        handler2.setRandomFactor(0)
        handler2.calcCPC()
        result_2 = handler2.getNetworkWithCPC()
        sym2 = handler2.calc_symmetry()

        pairs_nodes = []
        for node in result_1.nodes:
            pairs_nodes.append((result_1.nodes[node]['CPC'], result_2.nodes[node]['CPC']))
        pairs_edges = []
        for edge in result_1.edges:
            pairs_edges.append((result_1.edges[edge]['CPC'], result_2.edges[edge]['CPC']))

        # Extract values
        x, y = zip(*pairs_nodes)
        if np.std(x) == 0 or np.std(y) == 0:
            correlation_nodes = np.nan
        else:
            correlation_nodes,_ = pearsonr(x, y)

        # Extract values
        x, y = zip(*pairs_edges)
        if np.std(x) == 0 or np.std(y) == 0:
            correlation_edges = np.nan
        else:
            correlation_edges,_ = pearsonr(x, y)

        results.append((name, len(graph.nodes), len(graph.edges), T, sweep_value, correlation_nodes, correlation_edges, sym1, sym2))

        df = pd.DataFrame(results, columns=['name', 'number_of_nodes', 'number_of_edges', 'T', 'sweeps', 'node_corr', 'edge_corr', 'sym1', 'sym2'])
        dump(df, 'async_sync_results_AddHealth.joblib')

        # append pairs_nodes to node_log
        node_log.extend([(name, T, sweep_value, n1, n2) for n1, n2 in pairs_nodes])
        edge_log.extend([(name, T, sweep_value, e1, e2) for e1, e2 in pairs_edges])
        df_nodes_log = pd.DataFrame(node_log, columns=['name', 'T', 'sweeps', 'CPC_sync', 'CPC_async'])
        dump(df_nodes_log, 'async_sync_node_log_AddHealth.joblib')
        df_edges_log = pd.DataFrame(edge_log, columns=['name', 'T', 'sweeps', 'CPC_sync', 'CPC_async'])
        dump(df_edges_log, 'async_sync_edge_log_AddHealth.joblib')

print('finished')