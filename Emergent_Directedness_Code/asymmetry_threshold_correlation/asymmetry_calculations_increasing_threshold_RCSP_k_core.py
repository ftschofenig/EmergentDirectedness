import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
import pandas as pd
import seaborn as sns
import random
from itertools import product
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import traceback
import gc

from joblib import dump

import sys
import psutil
import os
import re

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))
import CPC_package as CPC

import seaborn as sns
import matplotlib.pyplot as plt

def log_ram():
    mem = psutil.Process(os.getpid()).memory_info().rss / 1e9
    print(f"RAM usage: {mem:.2f} GB")

np.seterr(invalid='ignore', divide='ignore')

CORES = 128
SWEEPS = 10
SEED_NP = 42
random.seed(SEED_NP)
np.random.seed(SEED_NP)

AddHealth_Graphs = CPC.getAddHealthGraphs()

####################################################################### RCS
if True:
    seed_function = CPC.randomFactorSeed
    T_values = [1, 2, 3, 4, 5] + [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    GRAPHS = AddHealth_Graphs
    CORRELATION_METHOD = 'pearson'

    results_threshold_asymmetry = []
    results_node_degree_cpc = []
    degree_cpc_tryout = []
    print('RCSP002_CA_asymmetry_threshold_AddHealth_k_core')
    progress_bar = tqdm(total=len(GRAPHS)*len(T_values))
    for name, graph in GRAPHS.items():
        #calculate once per graph
        nodes_outside = set(graph.nodes())
        try:
            betweenness = nx.betweenness_centrality(graph)
            closeness = nx.closeness_centrality(graph)
            eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
            kcore = nx.core_number(graph)
        except nx.NetworkXException as e:
            print(f"Centrality computation failed for {name} at T={T}: {e}")
            betweenness = closeness = eigenvector = {n: 0 for n in graph.nodes()}
            kcore = {n: 0 for n in graph.nodes()}

        for T in T_values:
            gc.collect()
            log_ram()
            progress_bar.update(1)
            handler = CPC.CpcHandler(graph, cores=CORES, seed_function=seed_function, sweeps=SWEEPS, model='GI', random_seed_np=SEED_NP)
            handler.to_dict_representation()
            handler.setThresholds(T)
            handler.setPortion(0.02)
            handler.setRandomFactor(0)
            handler.calcCPC()
            
            symmetry = handler.calc_symmetry()
            similarity = handler.calc_symmetry_cosine()

            spreadingDensity, steps = handler.getSpreadingDensity(with_steps=True)
            G = handler.getNetworkWithCPC()
            nodes_inside = set(G.nodes())
            assert nodes_outside == nodes_inside, f"Node sets do not match for graph {name} at T={T}"

            #create a df with degree of the nodes and CPC values of the nodes
            df_corr = pd.DataFrame()
            df_corr['degree'] = [d for n, d in G.degree()]
            df_corr['CPC'] = [G.nodes[n]['CPC'] for n in G.nodes()]
            correlation = 0#df_corr['CPC'].corr(df_corr['degree'], method=CORRELATION_METHOD)

            #calculate the cosine similarity between the degree and CPC values
            if np.all(df_corr['CPC'].values == 0) or np.all(df_corr['degree'].values == 0):
                cosine_correlation = np.nan
            else:
                cosine_correlation = cosine_similarity(df_corr['CPC'].values.reshape(1, -1), df_corr['degree'].values.reshape(1, -1))[0][0]

            df_corr['CPC_degree_normalized'] = df_corr['CPC'] / df_corr['degree'].replace(0, np.nan)

            normalized_correlation = 0#df_corr['CPC_degree_normalized'].corr(df_corr['degree'], method=CORRELATION_METHOD)

            if df_corr['CPC_degree_normalized'].isna().any() or np.all(df_corr['degree'].values == 0):
                cosine_normalized_correlation = np.nan
            else:
                cosine_normalized_correlation = cosine_similarity(df_corr['CPC_degree_normalized'].values.reshape(1, -1), df_corr['degree'].values.reshape(1, -1))[0][0]

            for node in G.nodes():
                degree_cpc_tryout.append((G.degree(node), betweenness[node], closeness[node], eigenvector[node], G.nodes[node]['CPC'], name, T, len(G.nodes()), spreadingDensity))

            temp = []
            for edge in G.edges():
                kcore_difference = kcore[edge[1]] - kcore[edge[0]]
                #cpc = G.edges[edge]['CPC']
                cpc = G.edges[edge]['CPC']-G.edges[edge[1], edge[0]]['CPC']
                temp.append((kcore_difference, cpc))
            df = pd.DataFrame(temp, columns=['kcore_difference', 'CPC'])

            if df['CPC'].std() == 0 or df['kcore_difference'].std() == 0:
                kcore_difference_cpc_correlation = np.nan
            else:
                kcore_difference_cpc_correlation = df['kcore_difference'].corr(df['CPC'], method=CORRELATION_METHOD)

            if np.all(df['kcore_difference'].values == 0) or np.all(df['CPC'].values == 0):
                cosine_kcore_difference_cpc_correlation = np.nan
            else:
                cosine_kcore_difference_cpc_correlation = cosine_similarity(df['kcore_difference'].values.reshape(1, -1), df['CPC'].values.reshape(1, -1))[0][0]

            results_threshold_asymmetry.append((name, T, len(G.nodes()), symmetry, similarity, correlation, cosine_correlation, normalized_correlation, cosine_normalized_correlation, spreadingDensity, kcore_difference_cpc_correlation, cosine_kcore_difference_cpc_correlation, steps))
            progress_bar.set_postfix({"T:": T, "name:": name, "number of nodes:": len(graph.nodes())})
        #create a dataframe with the results
        df = pd.DataFrame(results_threshold_asymmetry, columns=['name', 'T', 'Number_of_nodes','symmetry', 'similarity', 'node_cpc_degree_correlation', 'cosine_node_cpc_degree_correlation', 'normalized_correlation', 'cosine_normalized_correlation', 'spreadingDensity', 'Kcore_difference_CPC_correlation', 'Cosine_Kcore_difference_CPC_correlation', 'steps'])
        # Save the dictionary to a .joblib file
        dump(df, "./RCSP002_CA_asymmetry_threshold_AddHealth_kcore.joblib")
        df = pd.DataFrame(degree_cpc_tryout, columns=['degree', 'betweenness', 'closeness', 'eigenvector', 'CPC', 'name', 'T', 'Number_of_nodes', 'spreadingDensity'])
        dump(df, "./RCSP002_CA_kcore_cpc_AddHealth.joblib")