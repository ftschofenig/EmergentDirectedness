import warnings
warnings.simplefilter('error')  # Convert warnings to exceptions

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

CORES = 128
SWEEPS = 100

AddHealth_Graphs = CPC.getAddHealthGraphs()
Banerjee_graphs = CPC.getBanerjeeGraphs()
WS_graphs = []

for beta in np.linspace(0, 1, 100):
    for i in range(10):
        G = nx.watts_strogatz_graph(200, 8, beta)
        WS_graphs.append(G)

####################################################################### RCS
if True:
    seed_function = CPC.randomFactorSeed
    T_values = [1, 2, 3, 4, 5] + [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    GRAPHS = AddHealth_Graphs
    CORRELATION_METHOD = 'pearson'

    results_threshold_asymmetry = []
    results_node_degree_cpc = []
    degree_cpc_tryout = []
    print('RCSP002_CA_asymmetry_threshold_AddHealth')
    progress_bar = tqdm(total=len(GRAPHS)*len(T_values))
    for name, graph in GRAPHS.items():
        for T in T_values:
            gc.collect()
            log_ram()
            progress_bar.update(1)
            handler = CPC.CpcHandler(graph, cores=CORES, seed_function=seed_function, sweeps=SWEEPS, model='GI')
            handler.to_dict_representation()
            handler.setThresholds(T)
            handler.setPortion(0.02)
            handler.setRandomFactor(0)
            handler.calcCPC()
            
            symmetry = handler.calc_symmetry()
            similarity = handler.calc_symmetry_cosine()

            spreadingDensity, steps = handler.getSpreadingDensity(with_steps=True)
            G = handler.getNetworkWithCPC()

            try:
                #create a df with degree of the nodes and CPC values of the nodes
                df_corr = pd.DataFrame()
                df_corr['degree'] = [d for n, d in G.degree()]
                df_corr['CPC'] = [G.nodes[n]['CPC'] for n in G.nodes()]
                correlation = 0#df_corr['CPC'].corr(df_corr['degree'], method=CORRELATION_METHOD)
                #calculate the cosine similarity between the degree and CPC values
                cosine_correlation = cosine_similarity(df_corr['CPC'].values.reshape(1, -1), df_corr['degree'].values.reshape(1, -1))[0][0]

                df_corr['CPC_degree_normalized'] = df_corr['CPC']/df_corr['degree']
                
                normalized_correlation = 0#df_corr['CPC_degree_normalized'].corr(df_corr['degree'], method=CORRELATION_METHOD)
                cosine_normalized_correlation = cosine_similarity(df_corr['CPC_degree_normalized'].values.reshape(1, -1), df_corr['degree'].values.reshape(1, -1))[0][0]

                for node in G.nodes():
                    degree_cpc_tryout.append((G.degree(node), G.nodes[node]['CPC'], name, T, len(G.nodes()), spreadingDensity))

                temp = []
                for edge in G.edges():
                    degree_difference = G.degree(edge[1]) - G.degree(edge[0])
                    #cpc = G.edges[edge]['CPC']
                    cpc = G.edges[edge]['CPC']-G.edges[edge[1], edge[0]]['CPC']
                    temp.append((degree_difference, cpc))
                df = pd.DataFrame(temp, columns=['degree_difference', 'CPC'])
                degree_difference_cpc_correlation = df['degree_difference'].corr(df['CPC'], method=CORRELATION_METHOD)

                cosine_degree_difference_cpc_correlation = cosine_similarity(df['degree_difference'].values.reshape(1, -1), df['CPC'].values.reshape(1, -1))[0][0]

                results_threshold_asymmetry.append((name, T, len(G.nodes()), symmetry, similarity, correlation, cosine_correlation, normalized_correlation, cosine_normalized_correlation, spreadingDensity, degree_difference_cpc_correlation, cosine_degree_difference_cpc_correlation, steps))
                progress_bar.set_postfix({"T:": T, "name:": name, "number of nodes:": len(graph.nodes())})
            except Exception as e:
                print(f"[SKIP] Error for graph '{name}' at T={T}: {e}")
                traceback.print_exc()
            
                try:
                    # Diagnostic check for df_corr (node-level)
                    print("\n[DIAGNOSIS] Node-level correlation (df_corr):")
                    print("NaNs in 'degree':", df_corr['degree'].isna().sum())
                    print("NaNs in 'CPC':", df_corr['CPC'].isna().sum())
                    print("std(degree):", df_corr['degree'].std())
                    print("std(CPC):", df_corr['CPC'].std())
                    print("min/max(CPC):", df_corr['CPC'].min(), df_corr['CPC'].max())
            
                    # Diagnostic check for df (edge-level)
                    print("\n[DIAGNOSIS] Edge-level correlation (df):")
                    print("NaNs in 'degree_difference':", df['degree_difference'].isna().sum())
                    print("NaNs in 'CPC':", df['CPC'].isna().sum())
                    print("std(degree_difference):", df['degree_difference'].std())
                    print("std(CPC):", df['CPC'].std())
                    print("min/max(CPC):", df['CPC'].min(), df['CPC'].max())
                except Exception as diag_e:
                    print(f"[ERROR] Diagnostic check failed for '{name}' at T={T}: {diag_e}")
            
                continue
        #create a dataframe with the results
        df = pd.DataFrame(results_threshold_asymmetry, columns=['name', 'T', 'Number_of_nodes','symmetry', 'similarity', 'node_cpc_degree_correlation', 'cosine_node_cpc_degree_correlation', 'normalized_correlation', 'cosine_normalized_correlation', 'spreadingDensity', 'Degree_difference_CPC_correlation', 'Cosine_Degree_difference_CPC_correlation', 'steps'])
        # Save the dictionary to a .joblib file
        dump(df, "./RCSP002_CA_asymmetry_threshold_AddHealth.joblib")
        df = pd.DataFrame(degree_cpc_tryout, columns=['degree', 'CPC', 'name', 'T', 'Number_of_nodes', 'spreadingDensity'])
        dump(df, "./RCSP002_CA_degree_cpc_AddHealth.joblib")

if True:
    seed_function = CPC.randomFactorSeed
    T_values = [1, 2, 3, 4, 5] + [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    GRAPHS = Banerjee_graphs
    CORRELATION_METHOD = 'pearson'

    results_threshold_asymmetry = []
    results_node_degree_cpc = []
    degree_cpc_tryout = []
    print('RCSP002_CA_asymmetry_threshold_Banerjee')
    progress_bar = tqdm(total=len(GRAPHS)*len(T_values))
    for name, graph in GRAPHS.items():
        for T in T_values:
            gc.collect()
            log_ram()
            progress_bar.update(1)
            handler = CPC.CpcHandler(graph, cores=CORES, seed_function=seed_function, sweeps=SWEEPS, model='GI')
            handler.to_dict_representation()
            handler.setThresholds(T)
            handler.setPortion(0.02)
            handler.setRandomFactor(0)
            handler.calcCPC()
            
            symmetry = handler.calc_symmetry()
            similarity = handler.calc_symmetry_cosine()

            spreadingDensity, steps = handler.getSpreadingDensity(with_steps=True)
            G = handler.getNetworkWithCPC()

            try:
                #create a df with degree of the nodes and CPC values of the nodes
                df_corr = pd.DataFrame()
                df_corr['degree'] = [d for n, d in G.degree()]
                df_corr['CPC'] = [G.nodes[n]['CPC'] for n in G.nodes()]
                correlation = df_corr['CPC'].corr(df_corr['degree'], method=CORRELATION_METHOD)
                #calculate the cosine similarity between the degree and CPC values
                cosine_correlation = cosine_similarity(df_corr['CPC'].values.reshape(1, -1), df_corr['degree'].values.reshape(1, -1))[0][0]

                df_corr['CPC_degree_normalized'] = df_corr['CPC']/df_corr['degree']
                
                normalized_correlation = df_corr['CPC_degree_normalized'].corr(df_corr['degree'], method=CORRELATION_METHOD)
                cosine_normalized_correlation = cosine_similarity(df_corr['CPC_degree_normalized'].values.reshape(1, -1), df_corr['degree'].values.reshape(1, -1))[0][0]

                for node in G.nodes():
                    degree_cpc_tryout.append((G.degree(node), G.nodes[node]['CPC'], name, T, len(G.nodes()), spreadingDensity))

                temp = []
                for edge in G.edges():
                    degree_difference = G.degree(edge[1]) - G.degree(edge[0])
                    #cpc = G.edges[edge]['CPC']
                    cpc = G.edges[edge]['CPC']-G.edges[edge[1], edge[0]]['CPC']
                    temp.append((degree_difference, cpc))
                df = pd.DataFrame(temp, columns=['degree_difference', 'CPC'])
                degree_difference_cpc_correlation = df['degree_difference'].corr(df['CPC'], method=CORRELATION_METHOD)

                cosine_degree_difference_cpc_correlation = cosine_similarity(df['degree_difference'].values.reshape(1, -1), df['CPC'].values.reshape(1, -1))[0][0]

                results_threshold_asymmetry.append((name, T, len(G.nodes()), symmetry, similarity, correlation, cosine_correlation, normalized_correlation, cosine_normalized_correlation, spreadingDensity, degree_difference_cpc_correlation, cosine_degree_difference_cpc_correlation, steps))
                progress_bar.set_postfix({"T:": T, "name:": name, "number of nodes:": len(graph.nodes())})
            except Exception as e:
                print(f"[SKIP] Error for graph '{name}' at T={T}: {e}")
                traceback.print_exc()
            
                try:
                    # Diagnostic check for df_corr (node-level)
                    print("\n[DIAGNOSIS] Node-level correlation (df_corr):")
                    print("NaNs in 'degree':", df_corr['degree'].isna().sum())
                    print("NaNs in 'CPC':", df_corr['CPC'].isna().sum())
                    print("std(degree):", df_corr['degree'].std())
                    print("std(CPC):", df_corr['CPC'].std())
                    print("min/max(CPC):", df_corr['CPC'].min(), df_corr['CPC'].max())
            
                    # Diagnostic check for df (edge-level)
                    print("\n[DIAGNOSIS] Edge-level correlation (df):")
                    print("NaNs in 'degree_difference':", df['degree_difference'].isna().sum())
                    print("NaNs in 'CPC':", df['CPC'].isna().sum())
                    print("std(degree_difference):", df['degree_difference'].std())
                    print("std(CPC):", df['CPC'].std())
                    print("min/max(CPC):", df['CPC'].min(), df['CPC'].max())
                except Exception as diag_e:
                    print(f"[ERROR] Diagnostic check failed for '{name}' at T={T}: {diag_e}")
            
                continue
            
        #create a dataframe with the results
        df = pd.DataFrame(results_threshold_asymmetry, columns=['name', 'T', 'Number_of_nodes','symmetry', 'similarity', 'node_cpc_degree_correlation', 'cosine_node_cpc_degree_correlation', 'normalized_correlation', 'cosine_normalized_correlation', 'spreadingDensity', 'Degree_difference_CPC_correlation', 'Cosine_Degree_difference_CPC_correlation', 'steps'])
        # Save the dictionary to a .joblib file
        dump(df, "./RCSP002_CA_asymmetry_threshold_Banerjee.joblib")
        df = pd.DataFrame(degree_cpc_tryout, columns=['degree', 'CPC', 'name', 'T', 'Number_of_nodes', 'spreadingDensity'])
        dump(df, "./RCSP002_CA_degree_cpc_Banerjee.joblib")

if True:
    seed_function = CPC.randomFactorSeed
    T_values = [1, 2, 3, 4, 5] + [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    GRAPHS = WS_graphs
    CORRELATION_METHOD = 'pearson'

    results_threshold_asymmetry = []
    results_node_degree_cpc = []
    degree_cpc_tryout = []
    print('RCSP002_CA_asymmetry_threshold_WS')
    progress_bar = tqdm(total=len(GRAPHS)*len(T_values))
    name = 0
    for graph in GRAPHS:
        for T in T_values:
            gc.collect()
            log_ram()
            name += 1
            progress_bar.update(1)
            handler = CPC.CpcHandler(graph, cores=CORES, seed_function=seed_function, sweeps=SWEEPS, model='GI')
            handler.to_dict_representation()
            handler.setThresholds(T)
            handler.setPortion(0.02)
            handler.setRandomFactor(0)
            handler.calcCPC()
            
            symmetry = handler.calc_symmetry()
            similarity = handler.calc_symmetry_cosine()

            spreadingDensity, steps = handler.getSpreadingDensity(with_steps=True)
            G = handler.getNetworkWithCPC()

            try:
                #create a df with degree of the nodes and CPC values of the nodes
                df_corr = pd.DataFrame()
                df_corr['degree'] = [d for n, d in G.degree()]
                df_corr['CPC'] = [G.nodes[n]['CPC'] for n in G.nodes()]
                correlation = df_corr['CPC'].corr(df_corr['degree'], method=CORRELATION_METHOD)
                #calculate the cosine similarity between the degree and CPC values
                cosine_correlation = cosine_similarity(df_corr['CPC'].values.reshape(1, -1), df_corr['degree'].values.reshape(1, -1))[0][0]

                df_corr['CPC_degree_normalized'] = df_corr['CPC']/df_corr['degree']
                
                normalized_correlation = df_corr['CPC_degree_normalized'].corr(df_corr['degree'], method=CORRELATION_METHOD)
                cosine_normalized_correlation = cosine_similarity(df_corr['CPC_degree_normalized'].values.reshape(1, -1), df_corr['degree'].values.reshape(1, -1))[0][0]

                for node in G.nodes():
                    degree_cpc_tryout.append((G.degree(node), G.nodes[node]['CPC'], name, T, len(G.nodes()), spreadingDensity))

                temp = []
                for edge in G.edges():
                    degree_difference = G.degree(edge[1]) - G.degree(edge[0])
                    #cpc = G.edges[edge]['CPC']
                    cpc = G.edges[edge]['CPC']-G.edges[edge[1], edge[0]]['CPC']
                    temp.append((degree_difference, cpc))
                df = pd.DataFrame(temp, columns=['degree_difference', 'CPC'])
                degree_difference_cpc_correlation = df['degree_difference'].corr(df['CPC'], method=CORRELATION_METHOD)

                cosine_degree_difference_cpc_correlation = cosine_similarity(df['degree_difference'].values.reshape(1, -1), df['CPC'].values.reshape(1, -1))[0][0]

                results_threshold_asymmetry.append((name, T, len(G.nodes()), symmetry, similarity, correlation, cosine_correlation, normalized_correlation, cosine_normalized_correlation, spreadingDensity, degree_difference_cpc_correlation, cosine_degree_difference_cpc_correlation, steps))
                progress_bar.set_postfix({"T:": T, "name:": name, "number of nodes:": len(graph.nodes())})
            except Exception as e:
                print(f"[SKIP] Error for graph '{name}' at T={T}: {e}")
                traceback.print_exc()
            
                try:
                    # Diagnostic check for df_corr (node-level)
                    print("\n[DIAGNOSIS] Node-level correlation (df_corr):")
                    print("NaNs in 'degree':", df_corr['degree'].isna().sum())
                    print("NaNs in 'CPC':", df_corr['CPC'].isna().sum())
                    print("std(degree):", df_corr['degree'].std())
                    print("std(CPC):", df_corr['CPC'].std())
                    print("min/max(CPC):", df_corr['CPC'].min(), df_corr['CPC'].max())
            
                    # Diagnostic check for df (edge-level)
                    print("\n[DIAGNOSIS] Edge-level correlation (df):")
                    print("NaNs in 'degree_difference':", df['degree_difference'].isna().sum())
                    print("NaNs in 'CPC':", df['CPC'].isna().sum())
                    print("std(degree_difference):", df['degree_difference'].std())
                    print("std(CPC):", df['CPC'].std())
                    print("min/max(CPC):", df['CPC'].min(), df['CPC'].max())
                except Exception as diag_e:
                    print(f"[ERROR] Diagnostic check failed for '{name}' at T={T}: {diag_e}")
            
                continue
            
        #create a dataframe with the results
        df = pd.DataFrame(results_threshold_asymmetry, columns=['name', 'T', 'Number_of_nodes','symmetry', 'similarity', 'node_cpc_degree_correlation', 'cosine_node_cpc_degree_correlation', 'normalized_correlation', 'cosine_normalized_correlation', 'spreadingDensity', 'Degree_difference_CPC_correlation', 'Cosine_Degree_difference_CPC_correlation', 'steps'])
        # Save the dictionary to a .joblib file
        dump(df, "./RCSP002_CA_asymmetry_threshold_WS.joblib")
        df = pd.DataFrame(degree_cpc_tryout, columns=['degree', 'CPC', 'name', 'T', 'Number_of_nodes', 'spreadingDensity'])
        dump(df, "./RCSP002_CA_degree_cpc_WS.joblib")

####################################################################### RS
if True:
    seed_function = CPC.randomFactorSeed
    T_values = [1, 2, 3, 4, 5] + [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    GRAPHS = AddHealth_Graphs
    CORRELATION_METHOD = 'pearson'

    results_threshold_asymmetry = []
    results_node_degree_cpc = []
    degree_cpc_tryout = []
    print('RS002_CA_asymmetry_threshold_AddHealth')
    progress_bar = tqdm(total=len(GRAPHS)*len(T_values))
    for name, graph in GRAPHS.items():
        for T in T_values:
            gc.collect()
            log_ram()
            progress_bar.update(1)
            handler = CPC.CpcHandler(graph, cores=CORES, seed_function=seed_function, sweeps=SWEEPS, model='GI')
            handler.to_dict_representation()
            handler.setThresholds(T)
            handler.setPortion(0.02)
            handler.setRandomFactor(1)
            handler.calcCPC()
            
            symmetry = handler.calc_symmetry()
            similarity = handler.calc_symmetry_cosine()

            spreadingDensity, steps = handler.getSpreadingDensity(with_steps=True)
            G = handler.getNetworkWithCPC()

            try:
                #create a df with degree of the nodes and CPC values of the nodes
                df_corr = pd.DataFrame()
                df_corr['degree'] = [d for n, d in G.degree()]
                df_corr['CPC'] = [G.nodes[n]['CPC'] for n in G.nodes()]
                correlation = 0#df_corr['CPC'].corr(df_corr['degree'], method=CORRELATION_METHOD)
                #calculate the cosine similarity between the degree and CPC values
                cosine_correlation = cosine_similarity(df_corr['CPC'].values.reshape(1, -1), df_corr['degree'].values.reshape(1, -1))[0][0]

                df_corr['CPC_degree_normalized'] = df_corr['CPC']/df_corr['degree']
                
                normalized_correlation = 0#df_corr['CPC_degree_normalized'].corr(df_corr['degree'], method=CORRELATION_METHOD)
                cosine_normalized_correlation = cosine_similarity(df_corr['CPC_degree_normalized'].values.reshape(1, -1), df_corr['degree'].values.reshape(1, -1))[0][0]

                for node in G.nodes():
                    degree_cpc_tryout.append((G.degree(node), G.nodes[node]['CPC'], name, T, len(G.nodes()), spreadingDensity))

                temp = []
                for edge in G.edges():
                    degree_difference = G.degree(edge[1]) - G.degree(edge[0])
                    #cpc = G.edges[edge]['CPC']
                    cpc = G.edges[edge]['CPC']-G.edges[edge[1], edge[0]]['CPC']
                    temp.append((degree_difference, cpc))
                df = pd.DataFrame(temp, columns=['degree_difference', 'CPC'])
                degree_difference_cpc_correlation = df['degree_difference'].corr(df['CPC'], method=CORRELATION_METHOD)

                cosine_degree_difference_cpc_correlation = cosine_similarity(df['degree_difference'].values.reshape(1, -1), df['CPC'].values.reshape(1, -1))[0][0]

                results_threshold_asymmetry.append((name, T, len(G.nodes()), symmetry, similarity, correlation, cosine_correlation, normalized_correlation, cosine_normalized_correlation, spreadingDensity, degree_difference_cpc_correlation, cosine_degree_difference_cpc_correlation, steps))
                progress_bar.set_postfix({"T:": T, "name:": name, "number of nodes:": len(graph.nodes())})
            except Exception as e:
                print(f"[SKIP] Error for graph '{name}' at T={T}: {e}")
                traceback.print_exc()
            
                try:
                    # Diagnostic check for df_corr (node-level)
                    print("\n[DIAGNOSIS] Node-level correlation (df_corr):")
                    print("NaNs in 'degree':", df_corr['degree'].isna().sum())
                    print("NaNs in 'CPC':", df_corr['CPC'].isna().sum())
                    print("std(degree):", df_corr['degree'].std())
                    print("std(CPC):", df_corr['CPC'].std())
                    print("min/max(CPC):", df_corr['CPC'].min(), df_corr['CPC'].max())
            
                    # Diagnostic check for df (edge-level)
                    print("\n[DIAGNOSIS] Edge-level correlation (df):")
                    print("NaNs in 'degree_difference':", df['degree_difference'].isna().sum())
                    print("NaNs in 'CPC':", df['CPC'].isna().sum())
                    print("std(degree_difference):", df['degree_difference'].std())
                    print("std(CPC):", df['CPC'].std())
                    print("min/max(CPC):", df['CPC'].min(), df['CPC'].max())
                except Exception as diag_e:
                    print(f"[ERROR] Diagnostic check failed for '{name}' at T={T}: {diag_e}")
            
                continue
        #create a dataframe with the results
        df = pd.DataFrame(results_threshold_asymmetry, columns=['name', 'T', 'Number_of_nodes','symmetry', 'similarity', 'node_cpc_degree_correlation', 'cosine_node_cpc_degree_correlation', 'normalized_correlation', 'cosine_normalized_correlation', 'spreadingDensity', 'Degree_difference_CPC_correlation', 'Cosine_Degree_difference_CPC_correlation', 'steps'])
        # Save the dictionary to a .joblib file
        dump(df, "./RS002_CA_asymmetry_threshold_AddHealth.joblib")
        df = pd.DataFrame(degree_cpc_tryout, columns=['degree', 'CPC', 'name', 'T', 'Number_of_nodes', 'spreadingDensity'])
        dump(df, "./RS002_CA_degree_cpc_AddHealth.joblib")

if True:
    seed_function = CPC.randomFactorSeed
    T_values = [1, 2, 3, 4, 5] + [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    GRAPHS = Banerjee_graphs
    CORRELATION_METHOD = 'pearson'

    results_threshold_asymmetry = []
    results_node_degree_cpc = []
    degree_cpc_tryout = []
    print('RS002_CA_asymmetry_threshold_Banerjee')
    progress_bar = tqdm(total=len(GRAPHS)*len(T_values))
    for name, graph in GRAPHS.items():
        for T in T_values:
            gc.collect()
            log_ram()
            progress_bar.update(1)
            handler = CPC.CpcHandler(graph, cores=CORES, seed_function=seed_function, sweeps=SWEEPS, model='GI')
            handler.to_dict_representation()
            handler.setThresholds(T)
            handler.setPortion(0.02)
            handler.setRandomFactor(1)
            handler.calcCPC()
            
            symmetry = handler.calc_symmetry()
            similarity = handler.calc_symmetry_cosine()

            spreadingDensity, steps = handler.getSpreadingDensity(with_steps=True)
            G = handler.getNetworkWithCPC()

            try:
                #create a df with degree of the nodes and CPC values of the nodes
                df_corr = pd.DataFrame()
                df_corr['degree'] = [d for n, d in G.degree()]
                df_corr['CPC'] = [G.nodes[n]['CPC'] for n in G.nodes()]
                correlation = df_corr['CPC'].corr(df_corr['degree'], method=CORRELATION_METHOD)
                #calculate the cosine similarity between the degree and CPC values
                cosine_correlation = cosine_similarity(df_corr['CPC'].values.reshape(1, -1), df_corr['degree'].values.reshape(1, -1))[0][0]

                df_corr['CPC_degree_normalized'] = df_corr['CPC']/df_corr['degree']
                
                normalized_correlation = df_corr['CPC_degree_normalized'].corr(df_corr['degree'], method=CORRELATION_METHOD)
                cosine_normalized_correlation = cosine_similarity(df_corr['CPC_degree_normalized'].values.reshape(1, -1), df_corr['degree'].values.reshape(1, -1))[0][0]

                for node in G.nodes():
                    degree_cpc_tryout.append((G.degree(node), G.nodes[node]['CPC'], name, T, len(G.nodes()), spreadingDensity))

                temp = []
                for edge in G.edges():
                    degree_difference = G.degree(edge[1]) - G.degree(edge[0])
                    #cpc = G.edges[edge]['CPC']
                    cpc = G.edges[edge]['CPC']-G.edges[edge[1], edge[0]]['CPC']
                    temp.append((degree_difference, cpc))
                df = pd.DataFrame(temp, columns=['degree_difference', 'CPC'])
                degree_difference_cpc_correlation = df['degree_difference'].corr(df['CPC'], method=CORRELATION_METHOD)

                cosine_degree_difference_cpc_correlation = cosine_similarity(df['degree_difference'].values.reshape(1, -1), df['CPC'].values.reshape(1, -1))[0][0]

                results_threshold_asymmetry.append((name, T, len(G.nodes()), symmetry, similarity, correlation, cosine_correlation, normalized_correlation, cosine_normalized_correlation, spreadingDensity, degree_difference_cpc_correlation, cosine_degree_difference_cpc_correlation, steps))
                progress_bar.set_postfix({"T:": T, "name:": name, "number of nodes:": len(graph.nodes())})
            except Exception as e:
                print(f"[SKIP] Error for graph '{name}' at T={T}: {e}")
                traceback.print_exc()
            
                try:
                    # Diagnostic check for df_corr (node-level)
                    print("\n[DIAGNOSIS] Node-level correlation (df_corr):")
                    print("NaNs in 'degree':", df_corr['degree'].isna().sum())
                    print("NaNs in 'CPC':", df_corr['CPC'].isna().sum())
                    print("std(degree):", df_corr['degree'].std())
                    print("std(CPC):", df_corr['CPC'].std())
                    print("min/max(CPC):", df_corr['CPC'].min(), df_corr['CPC'].max())
            
                    # Diagnostic check for df (edge-level)
                    print("\n[DIAGNOSIS] Edge-level correlation (df):")
                    print("NaNs in 'degree_difference':", df['degree_difference'].isna().sum())
                    print("NaNs in 'CPC':", df['CPC'].isna().sum())
                    print("std(degree_difference):", df['degree_difference'].std())
                    print("std(CPC):", df['CPC'].std())
                    print("min/max(CPC):", df['CPC'].min(), df['CPC'].max())
                except Exception as diag_e:
                    print(f"[ERROR] Diagnostic check failed for '{name}' at T={T}: {diag_e}")
            
                continue
            
        #create a dataframe with the results
        df = pd.DataFrame(results_threshold_asymmetry, columns=['name', 'T', 'Number_of_nodes','symmetry', 'similarity', 'node_cpc_degree_correlation', 'cosine_node_cpc_degree_correlation', 'normalized_correlation', 'cosine_normalized_correlation', 'spreadingDensity', 'Degree_difference_CPC_correlation', 'Cosine_Degree_difference_CPC_correlation', 'steps'])
        # Save the dictionary to a .joblib file
        dump(df, "./RS002_CA_asymmetry_threshold_Banerjee.joblib")
        df = pd.DataFrame(degree_cpc_tryout, columns=['degree', 'CPC', 'name', 'T', 'Number_of_nodes', 'spreadingDensity'])
        dump(df, "./RS002_CA_degree_cpc_Banerjee.joblib")

if True:
    seed_function = CPC.randomFactorSeed
    T_values = [1, 2, 3, 4, 5] + [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    GRAPHS = WS_graphs
    CORRELATION_METHOD = 'pearson'

    results_threshold_asymmetry = []
    results_node_degree_cpc = []
    degree_cpc_tryout = []
    print('RS002_CA_asymmetry_threshold_WS')
    progress_bar = tqdm(total=len(GRAPHS)*len(T_values))
    name = 0
    for graph in GRAPHS:
        for T in T_values:
            gc.collect()
            log_ram()
            name += 1
            progress_bar.update(1)
            handler = CPC.CpcHandler(graph, cores=CORES, seed_function=seed_function, sweeps=SWEEPS, model='GI')
            handler.to_dict_representation()
            handler.setThresholds(T)
            handler.setPortion(0.02)
            handler.setRandomFactor(1)
            handler.calcCPC()
            
            symmetry = handler.calc_symmetry()
            similarity = handler.calc_symmetry_cosine()

            spreadingDensity, steps = handler.getSpreadingDensity(with_steps=True)
            G = handler.getNetworkWithCPC()

            try:
                #create a df with degree of the nodes and CPC values of the nodes
                df_corr = pd.DataFrame()
                df_corr['degree'] = [d for n, d in G.degree()]
                df_corr['CPC'] = [G.nodes[n]['CPC'] for n in G.nodes()]
                correlation = df_corr['CPC'].corr(df_corr['degree'], method=CORRELATION_METHOD)
                #calculate the cosine similarity between the degree and CPC values
                cosine_correlation = cosine_similarity(df_corr['CPC'].values.reshape(1, -1), df_corr['degree'].values.reshape(1, -1))[0][0]

                df_corr['CPC_degree_normalized'] = df_corr['CPC']/df_corr['degree']
                
                normalized_correlation = df_corr['CPC_degree_normalized'].corr(df_corr['degree'], method=CORRELATION_METHOD)
                cosine_normalized_correlation = cosine_similarity(df_corr['CPC_degree_normalized'].values.reshape(1, -1), df_corr['degree'].values.reshape(1, -1))[0][0]

                for node in G.nodes():
                    degree_cpc_tryout.append((G.degree(node), G.nodes[node]['CPC'], name, T, len(G.nodes()), spreadingDensity))

                temp = []
                for edge in G.edges():
                    degree_difference = G.degree(edge[1]) - G.degree(edge[0])
                    #cpc = G.edges[edge]['CPC']
                    cpc = G.edges[edge]['CPC']-G.edges[edge[1], edge[0]]['CPC']
                    temp.append((degree_difference, cpc))
                df = pd.DataFrame(temp, columns=['degree_difference', 'CPC'])
                degree_difference_cpc_correlation = df['degree_difference'].corr(df['CPC'], method=CORRELATION_METHOD)

                cosine_degree_difference_cpc_correlation = cosine_similarity(df['degree_difference'].values.reshape(1, -1), df['CPC'].values.reshape(1, -1))[0][0]

                results_threshold_asymmetry.append((name, T, len(G.nodes()), symmetry, similarity, correlation, cosine_correlation, normalized_correlation, cosine_normalized_correlation, spreadingDensity, degree_difference_cpc_correlation, cosine_degree_difference_cpc_correlation, steps))
                progress_bar.set_postfix({"T:": T, "name:": name, "number of nodes:": len(graph.nodes())})
            except Exception as e:
                print(f"[SKIP] Error for graph '{name}' at T={T}: {e}")
                traceback.print_exc()
            
                try:
                    # Diagnostic check for df_corr (node-level)
                    print("\n[DIAGNOSIS] Node-level correlation (df_corr):")
                    print("NaNs in 'degree':", df_corr['degree'].isna().sum())
                    print("NaNs in 'CPC':", df_corr['CPC'].isna().sum())
                    print("std(degree):", df_corr['degree'].std())
                    print("std(CPC):", df_corr['CPC'].std())
                    print("min/max(CPC):", df_corr['CPC'].min(), df_corr['CPC'].max())
            
                    # Diagnostic check for df (edge-level)
                    print("\n[DIAGNOSIS] Edge-level correlation (df):")
                    print("NaNs in 'degree_difference':", df['degree_difference'].isna().sum())
                    print("NaNs in 'CPC':", df['CPC'].isna().sum())
                    print("std(degree_difference):", df['degree_difference'].std())
                    print("std(CPC):", df['CPC'].std())
                    print("min/max(CPC):", df['CPC'].min(), df['CPC'].max())
                except Exception as diag_e:
                    print(f"[ERROR] Diagnostic check failed for '{name}' at T={T}: {diag_e}")
            
                continue
            
        #create a dataframe with the results
        df = pd.DataFrame(results_threshold_asymmetry, columns=['name', 'T', 'Number_of_nodes','symmetry', 'similarity', 'node_cpc_degree_correlation', 'cosine_node_cpc_degree_correlation', 'normalized_correlation', 'cosine_normalized_correlation', 'spreadingDensity', 'Degree_difference_CPC_correlation', 'Cosine_Degree_difference_CPC_correlation', 'steps'])
        # Save the dictionary to a .joblib file
        dump(df, "./RS002_CA_asymmetry_threshold_WS.joblib")
        df = pd.DataFrame(degree_cpc_tryout, columns=['degree', 'CPC', 'name', 'T', 'Number_of_nodes', 'spreadingDensity'])
        dump(df, "./RS_CA_degree_cpc_WS.joblib")