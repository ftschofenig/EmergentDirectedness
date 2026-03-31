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
np.seterr(invalid='ignore', divide='ignore')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

max_cores = multiprocessing.cpu_count()
print(f"Maximum number of cores available: {max_cores}")
AddHealth_Graphs = CPC.getAddHealthGraphs()
Banerjee_graphs = CPC.getBanerjeeGraphs()

CORES = 128 #max_cores
T_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 1, 2, 3]
sweep_value = 10
seed_function = CPC.randomFactorSeed
results = []
removal_portions = [0.1, 0.2, 0.3]

if True:
    print('Starting calculations on AddHealth...')
    tqdm_bar = tqdm(total=(len(AddHealth_Graphs)*len(T_values)*(len(removal_portions) + 1)))
    for name, graph in AddHealth_Graphs.items():
        original_edges_num = len(graph.edges())
        for T in T_values:
            G = graph.copy()
            handler = CPC.CpcHandler(G, cores=CORES, seed_function=seed_function, sweeps=sweep_value, random_seed_np=SEED)
            handler.to_dict_representation()
            handler.setThresholds(T)
            handler.setRandomFactor(0)
            handler.calcCPC()
            sym = handler.calc_symmetry()
            results.append((name, len(graph.nodes), original_edges_num, len(G.edges()), T, sweep_value, sym, 0.0))
            tqdm_bar.update(1)
            current_removed = 0
            for portion in removal_portions:
                num_total_remove = int(portion * original_edges_num)
                num_to_remove = num_total_remove - current_removed
                current_removed = num_total_remove
                if num_to_remove <= 0:
                    continue
                edges = list(G.edges())
                if num_to_remove >= len(edges):
                    break  # Cannot remove more than available
                edges_to_remove = random.sample(edges, num_to_remove)
                G.remove_edges_from(edges_to_remove)
                handler = CPC.CpcHandler(G, cores=CORES, seed_function=seed_function, sweeps=sweep_value, random_seed_np=SEED)
                handler.to_dict_representation()
                handler.setThresholds(T)
                handler.setRandomFactor(0)
                handler.calcCPC()
                sym = handler.calc_symmetry()
                results.append((name, len(graph.nodes), original_edges_num, len(G.edges()), T, sweep_value, sym, portion))
                tqdm_bar.update(1)
        df = pd.DataFrame(results, columns=['name', 'number_of_nodes', 'original_edges', 'edges_after', 'T', 'sweeps', 'symmetry', 'removal_portion'])
        dump(df, 'symmetry_stability_results_AddHealth.joblib')

if True:
    results = []
    print('Starting calculations on Banerjee...')
    tqdm_bar = tqdm(total=(len(Banerjee_graphs)*len(T_values)*(len(removal_portions) + 1)))
    for name, graph in Banerjee_graphs.items():
        original_edges_num = len(graph.edges())
        for T in T_values:
            G = graph.copy()
            handler = CPC.CpcHandler(G, cores=CORES, seed_function=seed_function, sweeps=sweep_value, random_seed_np=SEED)
            handler.to_dict_representation()
            handler.setThresholds(T)
            handler.setRandomFactor(0)
            handler.calcCPC()
            sym = handler.calc_symmetry()
            results.append((name, len(graph.nodes), original_edges_num, len(G.edges()), T, sweep_value, sym, 0.0))
            tqdm_bar.update(1)
            current_removed = 0
            for portion in removal_portions:
                num_total_remove = int(portion * original_edges_num)
                num_to_remove = num_total_remove - current_removed
                current_removed = num_total_remove
                if num_to_remove <= 0:
                    continue
                edges = list(G.edges())
                if num_to_remove >= len(edges):
                    break  # Cannot remove more than available
                edges_to_remove = random.sample(edges, num_to_remove)
                G.remove_edges_from(edges_to_remove)
                handler = CPC.CpcHandler(G, cores=CORES, seed_function=seed_function, sweeps=sweep_value, random_seed_np=SEED)
                handler.to_dict_representation()
                handler.setThresholds(T)
                handler.setRandomFactor(0)
                handler.calcCPC()
                sym = handler.calc_symmetry()
                results.append((name, len(graph.nodes), original_edges_num, len(G.edges()), T, sweep_value, sym, portion))
                tqdm_bar.update(1)
        df = pd.DataFrame(results, columns=['name', 'number_of_nodes', 'original_edges', 'edges_after', 'T', 'sweeps', 'symmetry', 'removal_portion'])
        dump(df, 'symmetry_stability_results_Banerjee.joblib')