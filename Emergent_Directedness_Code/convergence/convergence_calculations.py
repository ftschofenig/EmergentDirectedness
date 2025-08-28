import numpy as np
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
import pandas as pd
import seaborn as sns
import random
from itertools import product
import networkx as nx

from joblib import dump

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))
import CPC_package as CPC

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

from scipy.stats import pearsonr

import os
import networkx as nx
import pandas as pd
import re

import multiprocessing

max_cores = multiprocessing.cpu_count()
print(f"Maximum number of cores available: {max_cores}")

AddHealth_Graphs = CPC.getAddHealthGraphs()
Banerjee_graphs = CPC.getBanerjeeGraphs()

CORES = 128 #max_cores

T_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 1, 2, 3]

num_sweeps_values = [1, 2, 4, 8]

seed_function = CPC.randomFactorSeed

results = []
print('Starting calculations on AddHealth...')
tqdm_bar = tqdm(total=(len(AddHealth_Graphs)*len(T_values)*len(num_sweeps_values)))
for name, graph in AddHealth_Graphs.items():
    for T in T_values:
        tqdm_bar.update(1)
        for sweep_value in num_sweeps_values:
            handler = CPC.CpcHandler(graph, cores=CORES, seed_function=seed_function, sweeps=sweep_value)
            handler.to_dict_representation()
            handler.setThresholds(T)
            handler.setRandomFactor(0)
            handler.calcCPC()
            result_1 = handler.getNetworkWithCPC()
            sym1 = handler.calc_symmetry()

            handler2 = CPC.CpcHandler(graph, cores=CORES, seed_function=seed_function, sweeps=sweep_value)
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
            correlation_nodes,_ = pearsonr(x, y)

            # Extract values
            x, y = zip(*pairs_edges)
            correlation_edges,_ = pearsonr(x, y)

            results.append((name, len(graph.nodes), len(graph.edges), T, sweep_value, correlation_nodes, correlation_edges, sym1, sym2))

            df = pd.DataFrame(results, columns=['name', 'number_of_nodes', 'number_of_edges', 'T', 'sweeps', 'node_corr', 'edge_corr', 'sym1', 'sym2'])
            dump(df, 'convergence_results_AddHealth.joblib')

results = []
print('Starting calculations on Banerjee...')
tqdm_bar = tqdm(total=(len(Banerjee_graphs)*len(T_values)*len(num_sweeps_values)))
for name, graph in Banerjee_graphs.items():
    for T in T_values:
        tqdm_bar.update(1)
        for sweep_value in num_sweeps_values:
            handler = CPC.CpcHandler(graph, cores=CORES, seed_function=seed_function, sweeps=sweep_value)
            handler.to_dict_representation()
            handler.setThresholds(T)
            handler.setRandomFactor(0)
            handler.calcCPC()
            result_1 = handler.getNetworkWithCPC()
            sym1 = handler.calc_symmetry()

            handler2 = CPC.CpcHandler(graph, cores=CORES, seed_function=seed_function, sweeps=sweep_value)
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
            correlation_nodes,_ = pearsonr(x, y)

            # Extract values
            x, y = zip(*pairs_edges)
            correlation_edges,_ = pearsonr(x, y)

            results.append((name, len(graph.nodes), len(graph.edges), T, sweep_value, correlation_nodes, correlation_edges, sym1, sym2))

            df = pd.DataFrame(results, columns=['name', 'number_of_nodes', 'number_of_edges', 'T', 'sweeps', 'node_corr', 'edge_corr', 'sym1', 'sym2'])
            dump(df, 'convergence_results_Banerjee.joblib')