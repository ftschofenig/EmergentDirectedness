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
from sklearn.metrics.pairwise import cosine_similarity

CORES = 128
RUNS = 10
SWEEPS = 100

if True:
    p_values = [0] + [2**(-i) for i in range(40)] + np.arange(0, 1, 0.001).tolist() + np.arange(0, 0.1, 0.0001).tolist()
    #randomly shuffle the p_values
    random.shuffle(p_values)
    print(p_values)
    reps = range(RUNS)
    results = []
    for p, rep in tqdm(product(p_values, reps), total=len(p_values)*len(reps)):
        G = nx.watts_strogatz_graph(200, 8, 0)

        G = CPC.randomRewireGraphEdges(G, p)

        asymmetry_handler = CPC.CpcHandler(G.copy(), cores=CORES, seed_function=CPC.randomFactorSeed, sweeps=SWEEPS)
        asymmetry_handler.to_dict_representation()
        asymmetry_handler.setThresholds(2)
        asymmetry_handler.setRandomFactor(1)
        asymmetry_handler.setPortion(0.02)

        asymmetry_handler.calcCPC(tqdm_bar=False)
        cur_spread_density, cur_step_avg = asymmetry_handler.getSpreadingDensity(with_steps = True)

        asymmetry_handler.calc_symmetry_delta()

        G = asymmetry_handler.getNetworkWithCPC()
        all_CPC_values = []

        #sum all edge deltas and divide by the number of edges
        delta_sum = 0
        for edge in list(G.edges()):
            delta_sum += G.edges[edge]['delta']
            all_CPC_values.append(G.edges[edge]['CPC'])
        delta_sum = delta_sum/len(G.edges())


        results.append((p,asymmetry_handler.calc_symmetry(), cur_spread_density, cur_step_avg, 0, delta_sum, np.std(all_CPC_values)))
        # Save the results list to a .joblib file
        dump(results, "./RS002_CA_WS_200_8_T2_piece_dip_results.joblib")


if True:
    p_values = [0] + [2**(-i) for i in range(40)] + np.arange(0, 1, 0.001).tolist() + np.arange(0, 0.1, 0.0001).tolist()

    random.shuffle(p_values)
    print(p_values)
    reps = range(RUNS)
    results = []
    for p, rep in tqdm(product(p_values, reps), total=len(p_values)*len(reps)):
        G = nx.watts_strogatz_graph(200, 12, 0)

        G = CPC.randomRewireGraphEdges(G, p)

        asymmetry_handler = CPC.CpcHandler(G.copy(), cores=CORES, seed_function=CPC.randomFactorSeed, sweeps=SWEEPS)
        asymmetry_handler.to_dict_representation()
        asymmetry_handler.setThresholds(3)
        asymmetry_handler.setRandomFactor(1)
        asymmetry_handler.setPortion(0.03)

        asymmetry_handler.calcCPC(tqdm_bar=False)
        cur_spread_density, cur_step_avg = asymmetry_handler.getSpreadingDensity(with_steps = True)

        asymmetry_handler.calc_symmetry_delta()

        G = asymmetry_handler.getNetworkWithCPC()
        all_CPC_values = []

        #sum all edge deltas and divide by the number of edges
        delta_sum = 0
        for edge in list(G.edges()):
            delta_sum += G.edges[edge]['delta']
            all_CPC_values.append(G.edges[edge]['CPC'])
        delta_sum = delta_sum/len(G.edges())


        results.append((p,asymmetry_handler.calc_symmetry(), cur_spread_density, cur_step_avg, 0, delta_sum, np.std(all_CPC_values)))
        # Save the results list to a .joblib file
        dump(results, "./RS003_CA_WS_200_12_T3_piece_dip_results.joblib")