import networkx as nx
import random
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from tqdm import tqdm_notebook as tqdm
from matplotlib.patches import Patch

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
import pandas as pd
import seaborn as sns
import random
from itertools import product
from itertools import chain

from joblib import dump, load

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../../")))
import CPC_package as CPC

import re

CORES = 64
SWEEPS = 5

AddHealth_Graphs = CPC.getAddHealthGraphs()
Banerjee_graphs = CPC.getBanerjeeGraphs()

k_values = [2,3,4,5]
T_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]

if True:
    print("Powerlaw started")
    results = []
    for T in T_values:
        for k in [2,3,4,5]:
            print(k)
            for cluster_prob in tqdm(np.linspace(0, 1, len(AddHealth_Graphs.values()))):
                graph = nx.powerlaw_cluster_graph(1000, k, cluster_prob)
                handler = CPC.CpcHandler(graph.copy(), cores=CORES, seed_function=CPC.randomFactorSeed, sweeps = SWEEPS, model='GI')
                handler.setPortion(portion=0.02)
                handler.setRandomFactor(0)
                handler.to_dict_representation()
                handler.setThresholds(T)
                handler.calcCPC()
                G_after = handler.getNetworkWithCPC()
                G_after = CPC.calc_tie_ranges(G_after)

                #loop over all edges
                for edge in G_after.edges():
                    #get the CPC of the edge but also the CPC of the edge in the opposite direction
                    cpc = G_after.edges[edge]['CPC']
                    cpc_opposite = G_after.edges[(edge[1], edge[0])]['CPC']

                    #get the Tie Range of the edge
                    tie_range = G_after.edges[edge]['Tie_range']

                    #neighborhood overlap
                    #get the neighbors of the nodes of the edge
                    neighbors1 = set(G_after.neighbors(edge[0]))
                    neighbors2 = set(G_after.neighbors(edge[1]))

                    #degree delta
                    degree_delta = abs(G_after.degree[edge[0]] - G_after.degree[edge[1]])

                    #degree sum
                    degree_sum = G_after.degree[edge[0]] + G_after.degree[edge[1]]

                    #calculate the neighborhood overlap
                    neighborhood_overlap = len(neighbors1.intersection(neighbors2))
                    neighborhood_overlap_relative = neighborhood_overlap / (len(neighbors1) + len(neighbors2) - neighborhood_overlap - 2)
                    
                    #check if the values actually make sense and are not nan or inf
                    if tie_range is not None:
                        #add the result to the results dict
                        results.append((T, 'Power_law', str(k)+':'+str(cluster_prob),max(cpc,cpc_opposite),cpc+cpc_opposite,abs(cpc-cpc_opposite), tie_range, neighborhood_overlap, neighborhood_overlap_relative, degree_delta, degree_sum))

                #create a df from results
                df = pd.DataFrame(results, columns=['T', 'Type','ID','CPC_max','CPC_sum', 'CPC_difference', 'Tie_range', 'neighborhood_overlap', 'neighborhood_overlap_relative', 'degree_delta', 'degree_sum'])
                df['neighborhood_overlap_relative_ranked'] = df['neighborhood_overlap_relative'].rank(method='first')
                df['neighborhood_overlap_relative_quantile'] = pd.qcut(
                    df['neighborhood_overlap_relative_ranked'], 
                    3, 
                    labels=['weak', 'medium', 'strong']
                )
                dump(df, 'df_clustered_power_law.joblib')

if True:
    print("AddHealth started")
    results = []
    counter = 0
    for filename, graph in tqdm(AddHealth_Graphs.items()):
            for T in T_values:
                handler = CPC.CpcHandler(graph.copy(), cores=CORES, seed_function=CPC.randomFactorSeed, sweeps = SWEEPS, model='GI')
                handler.setPortion(portion=0.02)
                handler.setRandomFactor(0)
                handler.to_dict_representation()
                handler.setThresholds(T)
                handler.calcCPC()
                G_after = handler.getNetworkWithCPC()
                G_after = CPC.calc_tie_ranges(G_after)

                #loop over all edges
                for edge in G_after.edges():
                    #get the CPC of the edge but also the CPC of the edge in the opposite direction
                    cpc = G_after.edges[edge]['CPC']
                    cpc_opposite = G_after.edges[(edge[1], edge[0])]['CPC']

                    #get the Tie Range of the edge
                    tie_range = G_after.edges[edge]['Tie_range']

                    #neighborhood overlap
                    #get the neighbors of the nodes of the edge
                    neighbors1 = set(G_after.neighbors(edge[0]))
                    neighbors2 = set(G_after.neighbors(edge[1]))

                    #degree delta
                    degree_delta = abs(G_after.degree[edge[0]] - G_after.degree[edge[1]])

                    #degree sum
                    degree_sum = G_after.degree[edge[0]] + G_after.degree[edge[1]]

                    #calculate the neighborhood overlap
                    neighborhood_overlap = len(neighbors1.intersection(neighbors2))
                    neighborhood_overlap_relative = neighborhood_overlap / (len(neighbors1) + len(neighbors2) - neighborhood_overlap - 2)
                    
                    #check if the values actually make sense and are not nan or inf
                    if tie_range is not None:
                        #add the result to the results dict
                        results.append((T, 'AddHealth', filename, max(cpc,cpc_opposite),cpc+cpc_opposite,abs(cpc-cpc_opposite), tie_range, neighborhood_overlap, neighborhood_overlap_relative, degree_delta, degree_sum))

                #create a df from results
                df = pd.DataFrame(results, columns=['T','Type','ID','CPC_max','CPC_sum', 'CPC_difference', 'Tie_range', 'neighborhood_overlap', 'neighborhood_overlap_relative', 'degree_delta', 'degree_sum'])
                df['neighborhood_overlap_relative_ranked'] = df['neighborhood_overlap_relative'].rank(method='first')
                df['neighborhood_overlap_relative_quantile'] = pd.qcut(
                    df['neighborhood_overlap_relative_ranked'], 
                    3, 
                    labels=['weak', 'medium', 'strong']
                )
                dump(df, 'AddHealth.joblib')

if True:
    print("Banerjee started")
    results = []
    counter = 0
    for filename, graph in tqdm(Banerjee_graphs.items()):
        for T in T_values:
            handler = CPC.CpcHandler(graph.copy(), cores=CORES, seed_function=CPC.randomFactorSeed, sweeps = SWEEPS, model='GI')
            handler.setPortion(portion=0.02)
            handler.setRandomFactor(0)
            handler.to_dict_representation()
            handler.setThresholds(T)
            handler.calcCPC()
            G_after = handler.getNetworkWithCPC()
            G_after = CPC.calc_tie_ranges(G_after)

            #loop over all edges
            for edge in G_after.edges():
                #get the CPC of the edge but also the CPC of the edge in the opposite direction
                cpc = G_after.edges[edge]['CPC']
                cpc_opposite = G_after.edges[(edge[1], edge[0])]['CPC']

                #get the Tie Range of the edge
                tie_range = G_after.edges[edge]['Tie_range']

                #neighborhood overlap
                #get the neighbors of the nodes of the edge
                neighbors1 = set(G_after.neighbors(edge[0]))
                neighbors2 = set(G_after.neighbors(edge[1]))

                #degree delta
                degree_delta = abs(G_after.degree[edge[0]] - G_after.degree[edge[1]])

                #degree sum
                degree_sum = G_after.degree[edge[0]] + G_after.degree[edge[1]]

                #calculate the neighborhood overlap
                neighborhood_overlap = len(neighbors1.intersection(neighbors2))
                neighborhood_overlap_relative = neighborhood_overlap / (len(neighbors1) + len(neighbors2) - neighborhood_overlap - 2)
                
                #check if the values actually make sense and are not nan or inf
                if tie_range is not None:
                    #add the result to the results dict
                    results.append((T, 'Banerjee', filename,max(cpc,cpc_opposite),cpc+cpc_opposite,abs(cpc-cpc_opposite), tie_range, neighborhood_overlap, neighborhood_overlap_relative, degree_delta, degree_sum))

            #create a df from results
            df = pd.DataFrame(results, columns=['T','Type','ID','CPC_max','CPC_sum', 'CPC_difference', 'Tie_range', 'neighborhood_overlap', 'neighborhood_overlap_relative', 'degree_delta', 'degree_sum'])
            df['neighborhood_overlap_relative_ranked'] = df['neighborhood_overlap_relative'].rank(method='first')
            df['neighborhood_overlap_relative_quantile'] = pd.qcut(
                df['neighborhood_overlap_relative_ranked'], 
                3, 
                labels=['weak', 'medium', 'strong']
            )
            dump(df, 'Banerjee.joblib')

df_clustered_power_law = load("df_clustered_power_law.joblib")
df_AddHealth = load("AddHealth.joblib")
df_Banerjee = load("Banerjee.joblib")

#save to csv 
df_clustered_power_law.to_csv("df_clustered_power_law.csv")
df_AddHealth.to_csv("df_AddHealth.csv")
df_Banerjee.to_csv("df_Banerjee.csv")
df_all = pd.concat([df_clustered_power_law, df_AddHealth, df_Banerjee])
df_all = df_all.reset_index(drop=True)
df_all.to_csv("df_all.csv")