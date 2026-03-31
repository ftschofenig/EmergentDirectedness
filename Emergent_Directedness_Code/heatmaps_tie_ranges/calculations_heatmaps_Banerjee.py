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


from joblib import dump

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))
import CPC_package as CPC

AddHealth_Graphs = CPC.getAddHealthGraphs()
Banerjee_graphs = CPC.getBanerjeeGraphs()

SWEEPS = 5
CORES = 64
T = 2
repeat = 50
SEED_NP = 19
random.seed(SEED_NP)
np.random.seed(SEED_NP)

if True:
    print('Banerjee started')
    results = []
    for filename, graph in tqdm(Banerjee_graphs.items()):
        for rep in range(repeat):
            handler = CPC.CpcHandler(graph.copy(), cores=CORES, seed_function=CPC.randomFactorSeed, sweeps=SWEEPS, model='GI', random_seed_np=(SEED_NP+rep))
            handler.setPortion(portion=0.05)
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
                denominator = (len(neighbors1) + len(neighbors2) - neighborhood_overlap - 2)
                neighborhood_overlap_relative = (neighborhood_overlap / denominator) if denominator > 0 else np.nan
                
                #check if the values actually make sense and are not nan or inf
                if tie_range is not None:
                    #add the result to the results dict
                    results.append((max(cpc,cpc_opposite),cpc+cpc_opposite,abs(cpc-cpc_opposite), tie_range, neighborhood_overlap, neighborhood_overlap_relative, degree_delta, degree_sum))
    #create a df from results
    df = pd.DataFrame(results, columns=['CPC_max','CPC_sum', 'CPC_difference', 'Tie_range', 'neighborhood_overlap', 'neighborhood_overlap_relative', 'degree_delta', 'degree_sum'])
    df['neighborhood_overlap_relative_ranked'] = df['neighborhood_overlap_relative'].rank(method='first')
    df['neighborhood_overlap_relative_quantile'] = pd.qcut(
        df['neighborhood_overlap_relative_ranked'], 
        3, 
        labels=['weak', 'medium', 'strong']
    )

    dump(df, f'df_Banerjee.joblib')

print('finished')