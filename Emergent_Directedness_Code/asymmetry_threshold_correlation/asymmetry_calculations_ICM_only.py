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

CORES = 64
SWEEPS = 10

AddHealth_Graphs = CPC.getAddHealthGraphs()
Banerjee_graphs = CPC.getBanerjeeGraphs()


seed_function = CPC.randomFactorSeed

GRAPHS = AddHealth_Graphs
CORRELATION_METHOD = 'pearson'
edge_probabilities = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

results_threshold_asymmetry = []
results_node_degree_cpc = []
degree_cpc_tryout = []
print('ICM_RCSP002_CA_asymmetry_threshold_AddHealth')
progress_bar = tqdm(total=len(GRAPHS)*len(edge_probabilities))

for name, graph in GRAPHS.items():
    for eP in edge_probabilities:
        progress_bar.update(1)
        handler = CPC.CpcHandler(graph, cores=CORES, seed_function=seed_function, sweeps=SWEEPS, model='ICM')
        handler.to_dict_representation()
        handler.setPortion(0.02)
        handler.setRandomFactor(0)
        handler.setProbabilitiesICM(eP)
        handler.setThresholds(1)
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
                degree_cpc_tryout.append((G.degree(node), G.nodes[node]['CPC'], name, eP, len(G.nodes()), spreadingDensity))

            temp = []
            for edge in G.edges():
                degree_difference = G.degree(edge[1]) - G.degree(edge[0])
                #cpc = G.edges[edge]['CPC']
                cpc = G.edges[edge]['CPC']-G.edges[edge[1], edge[0]]['CPC']
                temp.append((degree_difference, cpc))
            df = pd.DataFrame(temp, columns=['degree_difference', 'CPC'])
            degree_difference_cpc_correlation = df['degree_difference'].corr(df['CPC'], method=CORRELATION_METHOD)

            cosine_degree_difference_cpc_correlation = cosine_similarity(df['degree_difference'].values.reshape(1, -1), df['CPC'].values.reshape(1, -1))[0][0]

            results_threshold_asymmetry.append((name, eP, len(G.nodes()), symmetry, similarity, correlation, cosine_correlation, normalized_correlation, cosine_normalized_correlation, spreadingDensity, degree_difference_cpc_correlation, cosine_degree_difference_cpc_correlation, steps))
            progress_bar.set_postfix({"eP:": eP, "name:": name, "number of nodes:": len(graph.nodes())})
        except Exception as e:
                    print(f"[SKIP] Error for graph '{name}' at eP={eP}: {e}")
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
                        print(f"[ERROR] Diagnostic check failed for '{name}' at eP={eP}: {diag_e}")
                
                    continue
        
    #create a dataframe with the results
    df = pd.DataFrame(results_threshold_asymmetry, columns=['name', 'eP', 'Number_of_nodes','symmetry', 'similarity', 'node_cpc_degree_correlation', 'cosine_node_cpc_degree_correlation', 'normalized_correlation', 'cosine_normalized_correlation', 'spreadingDensity', 'Degree_difference_CPC_correlation', 'Cosine_Degree_difference_CPC_correlation', 'steps'])
    # Save the dictionary to a .joblib file
    dump(df, "./ICM_RCSP002_CA_asymmetry_threshold_AddHealth.joblib")
    df = pd.DataFrame(degree_cpc_tryout, columns=['degree', 'CPC', 'name', 'eP', 'Number_of_nodes', 'spreadingDensity'])
    dump(df, "./ICM_RCSP002_CA_degree_cpc_AddHealth.joblib")

results_threshold_asymmetry = []
results_node_degree_cpc = []
degree_cpc_tryout = []
print('ICM_RS002_CA_asymmetry_threshold_AddHealth')
progress_bar = tqdm(total=len(GRAPHS)*len(edge_probabilities))

for name, graph in GRAPHS.items():
    for eP in edge_probabilities:
        progress_bar.update(1)
        handler = CPC.CpcHandler(graph, cores=CORES, seed_function=seed_function, sweeps=SWEEPS, model='ICM')
        handler.to_dict_representation()
        handler.setPortion(0.02)
        handler.setRandomFactor(1)
        handler.setProbabilitiesICM(eP)
        handler.setThresholds(-1)
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
                degree_cpc_tryout.append((G.degree(node), G.nodes[node]['CPC'], name, eP, len(G.nodes()), spreadingDensity))

            temp = []
            for edge in G.edges():
                degree_difference = G.degree(edge[1]) - G.degree(edge[0])
                #cpc = G.edges[edge]['CPC']
                cpc = G.edges[edge]['CPC']-G.edges[edge[1], edge[0]]['CPC']
                temp.append((degree_difference, cpc))
            df = pd.DataFrame(temp, columns=['degree_difference', 'CPC'])
            degree_difference_cpc_correlation = df['degree_difference'].corr(df['CPC'], method=CORRELATION_METHOD)

            cosine_degree_difference_cpc_correlation = cosine_similarity(df['degree_difference'].values.reshape(1, -1), df['CPC'].values.reshape(1, -1))[0][0]

            results_threshold_asymmetry.append((name, eP, len(G.nodes()), symmetry, similarity, correlation, cosine_correlation, normalized_correlation, cosine_normalized_correlation, spreadingDensity, degree_difference_cpc_correlation, cosine_degree_difference_cpc_correlation, steps))
            progress_bar.set_postfix({"eP:": eP, "name:": name, "number of nodes:": len(graph.nodes())})
        except Exception as e:
                print(f"[SKIP] Error for graph '{name}' at eP={eP}: {e}")
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
                    print(f"[ERROR] Diagnostic check failed for '{name}' at eP={eP}: {diag_e}")
            
                continue
        
    #create a dataframe with the results
    df = pd.DataFrame(results_threshold_asymmetry, columns=['name', 'eP', 'Number_of_nodes','symmetry', 'similarity', 'node_cpc_degree_correlation', 'cosine_node_cpc_degree_correlation', 'normalized_correlation', 'cosine_normalized_correlation', 'spreadingDensity', 'Degree_difference_CPC_correlation', 'Cosine_Degree_difference_CPC_correlation', 'steps'])
    # Save the dictionary to a .joblib file
    dump(df, "./ICM_RS002_CA_asymmetry_threshold_AddHealth.joblib")
    df = pd.DataFrame(degree_cpc_tryout, columns=['degree', 'CPC', 'name', 'eP', 'Number_of_nodes', 'spreadingDensity'])
    dump(df, "./ICM_RS002_CA_degree_cpc_AddHealth.joblib")
    