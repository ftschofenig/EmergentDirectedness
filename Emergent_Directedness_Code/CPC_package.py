import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
import pandas as pd
import seaborn as sns
import random
from sklearn.metrics.pairwise import cosine_similarity
import math
import sys
import os
import re


def getAddHealthGraphs(directory = '../../AddHealth_Networks_Largest_Components/'):
    AddHealth_Graphs = {}
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            adj_matrix = pd.read_csv(filepath, header=None, skiprows=1)
            G = nx.from_pandas_adjacency(adj_matrix, create_using=nx.Graph)
            
            if len(G.nodes())>25:
                AddHealth_Graphs[filename] = G

    #sort the keys by increasing number of nodes
    AddHealth_Graphs = {k: v for k, v in sorted(AddHealth_Graphs.items(), key=lambda item: item[1].number_of_nodes())}
    return AddHealth_Graphs

def getBanerjeeGraphs(directory = '../../datav4.0/Data/1. Network Data/Adjacency Matrices/'):
    Banerjee_graphs = {}

    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            adj_matrix = pd.read_csv(filepath, header=None)
            G = nx.from_pandas_adjacency(adj_matrix)

            village_number = int(re.search(r'(\d+)\.csv$', filename).group(1))
            if village_number in Banerjee_graphs:
                Banerjee_graphs[village_number].append(G)
            else:
                Banerjee_graphs[village_number] = [G]

            #remove all nodes with degree 0 from the graph
            G.remove_nodes_from(list(nx.isolates(G)))

    Banerjee_graphs_keys = list(Banerjee_graphs.keys())
    #merge all graphs for each village
    for i in Banerjee_graphs_keys:
        #print(i, len(Banerjee_graphs[i]))
        if len(Banerjee_graphs[i]) > 0:
            Banerjee_graphs[i] = nx.compose_all(Banerjee_graphs[i])
            #cast the graph to a simple graph
            Banerjee_graphs[i] = nx.Graph(Banerjee_graphs[i])
        else:
            #remove the entry from the dictionary
            Banerjee_graphs.pop(i)

    for i in Banerjee_graphs:
        #remove all nodes with degree 0 from the graph
        Banerjee_graphs[i].remove_nodes_from(list(nx.isolates(Banerjee_graphs[i])))

    #sort the keys by increasing number of nodes
    Banerjee_graphs = {k: v for k, v in sorted(Banerjee_graphs.items(), key=lambda item: item[1].number_of_nodes(), reverse=False)}

    #filter all graphs with less than 25 nodes
    Banerjee_graphs = {k: v for k, v in Banerjee_graphs.items() if v.number_of_nodes() > 25}

    #drop all nodes which are not part of the giant component
    for i in Banerjee_graphs:
        Banerjee_graphs[i] = Banerjee_graphs[i].subgraph(max(nx.connected_components(Banerjee_graphs[i]), key=len)).copy()

    return Banerjee_graphs


def randomRewireGraphEdges(G, p):
    edges =  list(G.edges())

    for edge in edges:
        if random.random() < p:
            G.remove_edge(*edge)
            new_edge = random.sample(list(G.nodes()), 2)
            
            while new_edge in list(G.edges()):
                new_edge = random.sample(list(G.nodes()), 2)

            G.add_edge(*new_edge)
    
    return G

def randomAddGraphEdges(G, to_add = 10):
    G = nx.Graph(G)

    edges =  list(G.edges())

    for i in range(to_add):
        #randomly select two nodes
        node1 = random.choice(list(G.nodes()))
        node2 = random.choice(list(G.nodes()))
        while node1 == node2 or (node1, node2) in edges or (node2, node1) in edges:
            node2 = random.choice(list(G.nodes()))
        
        G.add_edge(node1, node2)

    G = nx.DiGraph(G)
    
    return G

def calc_tie_ranges(G):
    """Calculate the tie range (second shortest path length) for each edge in the graph."""
    G_copy = G.copy()
    for edge in list(G.edges()):  
        G_copy.remove_edge(*edge)
        try:
            G.edges[edge]['Tie_range'] = nx.shortest_path_length(G_copy, source=edge[0], target=edge[1])
        except Exception as e:
            G.edges[edge]['Tie_range'] = None
        G_copy.add_edge(*edge) 
    return G

def checkSeedingCapability(seed, adjacency_dict_successors, disabled_nodes, disabled_edges, activation_times):
    activation_times[seed] = 0
    list_of_nodes_that_became_active = {seed}
    
    for neib in adjacency_dict_successors[seed]: #take care, only successors are seeds this way
        if neib not in disabled_nodes and (seed, neib) not in disabled_edges:
            activation_times[neib] = 0
            list_of_nodes_that_became_active.add(neib)
    
    return activation_times, list_of_nodes_that_became_active


def t_minus_one_seeding(adjacency_dict_successors, disabled_nodes, disabled_edges, activation_times):
    if len(disabled_nodes) == len(list(adjacency_dict_successors.keys())):
        return activation_times, {}

    seed = np.random.choice(list(adjacency_dict_successors.keys()))
    while seed in disabled_nodes:
        seed = np.random.choice(list(adjacency_dict_successors.keys()))

    activation_times[seed] = 0
    list_of_nodes_that_became_active = {seed}
    for neib in adjacency_dict_successors[seed]: #take care, only successors are seeds this way
        if neib not in disabled_nodes and (seed, neib) not in disabled_edges:
            activation_times[neib] = 0
            list_of_nodes_that_became_active.add(neib)
    
    return activation_times, list_of_nodes_that_became_active

def randomFactorSeed(adjacency_dict_successors, disabled_nodes, disabled_edges, activation_times, seed_portion, randomFactor):
    list_of_nodes_that_became_active = set()
    number_of_nodes = len(adjacency_dict_successors.keys())
    number_of_seeds = np.ceil(seed_portion * number_of_nodes)

    if number_of_seeds > (len(list(adjacency_dict_successors.keys()))-len(disabled_nodes)):
        number_of_seeds = (len(list(adjacency_dict_successors.keys()))-len(disabled_nodes))

    while len(list_of_nodes_that_became_active) < number_of_seeds:
        if random.random() < randomFactor or len(list_of_nodes_that_became_active) == 0:
            seed = np.random.choice(list(adjacency_dict_successors.keys()))
            while seed in disabled_nodes:
                seed = np.random.choice(list(adjacency_dict_successors.keys()))
        else:
            possibleNeibs = []

            for node in list_of_nodes_that_became_active: #this is to make sure that the seed is connected to the already activated nodes
                possibleNeibs += adjacency_dict_successors[node]
            possibleNeibs = list(set(possibleNeibs))

            #check if there are even any possible neighbors which are not active or disabled
            possibleNeibs = [neib for neib in possibleNeibs if neib not in disabled_nodes and neib not in list_of_nodes_that_became_active]

            if len(possibleNeibs) == 0: #if there is no adjacent node available anymore
                possibleNeibs = list(adjacency_dict_successors.keys())

            seed = np.random.choice(possibleNeibs)
            while seed in disabled_nodes or seed in list_of_nodes_that_became_active:
                seed = np.random.choice(possibleNeibs)

        activation_times[seed] = 0
        list_of_nodes_that_became_active.add(seed)
    
    return activation_times, list_of_nodes_that_became_active

def calcCPCForNetwork(args):
    #the aim of this function is to calculate the CPC of a network
    #the network is represented by an adjacency list and an edge list
    #the threshold is passed as a parameter
    
    adjacency_dict, threshold_dict, number_of_seeds, disabled_nodes, disabled_edges, adjacency_dict_successors, seeding_function, tqdm_bar, always_active_set, random_portion, randomFactor, model, edgeWeights, probability_noisy, icm_probabilities = args

    causal_ambiguity = []
    
    #first create an edge_dict and an node_dict_CPC
    edge_dict_CPC = {}
    node_dict_CPC = {}
    activation_times = {}
    active_neibs_at_activation = {}

    for node in adjacency_dict:
        for neighbour in adjacency_dict[node]:
            edge_dict_CPC[(neighbour, node)] = 0
        node_dict_CPC[node] = 0
        activation_times[node] = -1

    isConverged = False

    i = 0
    if tqdm_bar:
        progress_bar = tqdm(total=number_of_seeds)

    while not isConverged:
        i += 1
        
        if tqdm_bar:
            progress_bar.update(1)

        if i > number_of_seeds:
            isConverged = True 

        #reset all activation times
        for node in activation_times:
            activation_times[node] = -1
            active_neibs_at_activation[node] = 0

        #1. create seed
        if seeding_function is randomFactorSeed:
            activation_times, list_of_nodes_that_became_active = seeding_function(adjacency_dict_successors, disabled_nodes, disabled_edges, activation_times, random_portion, randomFactor)
        else:
            activation_times, list_of_nodes_that_became_active = seeding_function(adjacency_dict_successors, disabled_nodes, disabled_edges, activation_times)
        
        list_of_nodes_that_became_active = set(list_of_nodes_that_became_active)

        for node in always_active_set:
            activation_times[node] = 0

        list_of_nodes_that_became_active = set(list_of_nodes_that_became_active).union(always_active_set)
        temp_seed_nodes = list(list_of_nodes_that_became_active)

        #2. spread until convergence
        newActivation = True
        t = 0
        close_distance_approximation = 200

        if model == 'GI':
            while newActivation and t < close_distance_approximation:
                t += 1
                newActivation = False
                for node in activation_times.keys():
                    if activation_times[node] == -1:
                        activeNeighborCounter = 0
                        for predecessor in adjacency_dict[node]:
                            if activation_times[predecessor] < t and activation_times[predecessor] > -1 and (predecessor, node) not in disabled_edges: #these are the ones activated before
                                activeNeighborCounter += 1
                        if activeNeighborCounter >= threshold_dict[node]:
                            if node not in disabled_nodes:
                                activation_times[node] = t
                                newActivation = True
                                list_of_nodes_that_became_active.add(node)
                                causal_ambiguity.append(activeNeighborCounter/threshold_dict[node])
                                active_neibs_at_activation[node] = activeNeighborCounter
        elif model == 'NOISY':
            assert probability_noisy is not None

            while newActivation and t < close_distance_approximation:
                t += 1
                newActivation = False
                for node in activation_times.keys():
                    if activation_times[node] == -1:
                        activeNeighborCounter = 0
                        for predecessor in adjacency_dict[node]:
                            if activation_times[predecessor] < t and activation_times[predecessor] > -1 and (predecessor, node) not in disabled_edges: #these are the ones activated before
                                activeNeighborCounter += 1
                        if activeNeighborCounter >= threshold_dict[node]:
                            if node not in disabled_nodes:
                                activation_times[node] = t
                                newActivation = True
                                list_of_nodes_that_became_active.add(node)
                                causal_ambiguity.append(activeNeighborCounter/threshold_dict[node])
                                active_neibs_at_activation[node] = activeNeighborCounter

                        if activeNeighborCounter > 0 and activation_times[node] == -1:
                            if random.random() <= probability_noisy:
                                if node not in disabled_nodes:
                                    activation_times[node] = t
                                    newActivation = True
                                    list_of_nodes_that_became_active.add(node)
                                    causal_ambiguity.append(activeNeighborCounter/threshold_dict[node])
                                    active_neibs_at_activation[node] = activeNeighborCounter
        elif model == 'NOISY_SINGLE':
            assert probability_noisy is not None
            transmissions_tried = set()
            while newActivation and t < close_distance_approximation:
                t += 1
                newActivation = False
                for node in activation_times.keys():
                    if activation_times[node] == -1:
                        activeNeighborCounter = 0
                        activeNeighbors = set()
                        for predecessor in adjacency_dict[node]:
                            if activation_times[predecessor] < t and activation_times[predecessor] > -1 and (predecessor, node) not in disabled_edges: #these are the ones activated before
                                activeNeighborCounter += 1
                                activeNeighbors.add(predecessor)
                        if activeNeighborCounter >= threshold_dict[node]:
                            if node not in disabled_nodes:
                                activation_times[node] = t
                                newActivation = True
                                list_of_nodes_that_became_active.add(node)
                                causal_ambiguity.append(activeNeighborCounter/threshold_dict[node])
                                active_neibs_at_activation[node] = activeNeighborCounter

                        if activeNeighborCounter > 0 and activation_times[node] == -1:
                            if random.random() <= probability_noisy and node not in transmissions_tried:
                                if node not in disabled_nodes:
                                    activation_times[node] = t
                                    newActivation = True
                                    list_of_nodes_that_became_active.add(node)
                                    causal_ambiguity.append(activeNeighborCounter/threshold_dict[node])
                                    active_neibs_at_activation[node] = activeNeighborCounter
                            transmissions_tried.add(node)
        elif model == 'LTM':
            assert edgeWeights is not None

            while newActivation and t < close_distance_approximation:
                t += 1
                newActivation = False
                for node in activation_times.keys():
                    if activation_times[node] == -1:
                        activeNeighborCounter = 0
                        for predecessor in adjacency_dict[node]:
                            if activation_times[predecessor] < t and activation_times[predecessor] > -1 and (predecessor, node) not in disabled_edges: #these are the ones activated before
                                activeNeighborCounter += edgeWeights[(predecessor, node)] #this way it gets immediately weighted, otherwise its basically the GI model
                        if activeNeighborCounter >= threshold_dict[node]:
                            if node not in disabled_nodes:
                                activation_times[node] = t
                                newActivation = True
                                list_of_nodes_that_became_active.add(node)
                                causal_ambiguity.append(activeNeighborCounter/threshold_dict[node])
                                active_neibs_at_activation[node] = activeNeighborCounter
        elif model == 'ICM':
            assert icm_probabilities is not None

            usedEdges = set()
            while newActivation and t < close_distance_approximation:
                t += 1
                newActivation = False

                for node in activation_times.keys():
                    if activation_times[node] == -1:
                        for neib in adjacency_dict[node]:
                            if activation_times[neib] > -1 and activation_times[neib] < t and (neib, node) not in usedEdges:
                                usedEdges.add((neib, node))
                                if random.random() <= icm_probabilities[(neib,node)]:
                                    activation_times[node] = t
                                    newActivation = True
                                    list_of_nodes_that_became_active.add(node)
        else:
            raise NotImplementedError("Unknown Model")

        #3. for every activated node
        while len(list_of_nodes_that_became_active) > 0:
            cur_focus_node = list_of_nodes_that_became_active.pop()

            member_list = set([cur_focus_node])
            checked_list = set()
            edges_used = set()

            while len(member_list) > len(checked_list):
                newMembers = set()
                newChecked = set()
                for node in (member_list - checked_list):
                    newChecked.add(node)
                    
                    for predecessor in adjacency_dict[node]:
                        if activation_times[predecessor] < (activation_times[node]) and activation_times[predecessor] > -1:
                            newMembers.add(predecessor)
                            if (predecessor, node) not in disabled_edges:
                                edges_used.add((predecessor, node))

                member_list = member_list.union(newMembers)
                checked_list = checked_list.union(newChecked)

            #4. update the CPC values
            for node in member_list: 
                node_dict_CPC[node] += 1
            for edge in edges_used:
                edge_dict_CPC[edge] += 1
        
    return node_dict_CPC, edge_dict_CPC, causal_ambiguity

def calcSpreadingDensityForNetwork(args):
    np.random.seed()

    adjacency_dict, threshold_dict, number_of_seeds, disabled_nodes, disabled_edges, adjacency_dict_successors, seeding_function, random_portion, randomFactor, model, edgeWeights, probability_noisy, icm_probabilities = args

    spreading_densities = []

    activation_times = {}
    for node in adjacency_dict:
        activation_times[node] = -1

    isConverged = False

    i = 0

    number_of_steps = []

    while not isConverged:
        i += 1
        if i > number_of_seeds:
            isConverged = True 

        #reset all activation times
        for node in activation_times:
            activation_times[node] = -1

        #1. create seed
        if seeding_function is randomFactorSeed:
            activation_times, list_of_nodes_that_became_active = seeding_function(adjacency_dict_successors, disabled_nodes, disabled_edges, activation_times, random_portion, randomFactor)
        else:
            activation_times, list_of_nodes_that_became_active = seeding_function(adjacency_dict_successors, disabled_nodes, disabled_edges, activation_times)
        list_of_nodes_that_became_active = set(list_of_nodes_that_became_active)

        #2. spread until convergence
        newActivation = True
        t = 0
        close_distance_approximation = 200

        if model == 'GI':
            while newActivation and t < close_distance_approximation:
                t += 1
                newActivation = False
                for node in activation_times.keys():
                    if activation_times[node] == -1:
                        activeNeighborCounter = 0
                        for predecessor in adjacency_dict[node]:
                            if activation_times[predecessor] < t and activation_times[predecessor] > -1 and (predecessor, node) not in disabled_edges: #these are the ones activated before
                                activeNeighborCounter += 1
                        if activeNeighborCounter >= threshold_dict[node]:
                            if node not in disabled_nodes:
                                activation_times[node] = t
                                newActivation = True
                                list_of_nodes_that_became_active.add(node)
        elif model == 'NOISY':
             assert probability_noisy is not None

             while newActivation and t < close_distance_approximation:
                t += 1
                newActivation = False
                for node in activation_times.keys():
                    if activation_times[node] == -1:
                        activeNeighborCounter = 0
                        for predecessor in adjacency_dict[node]:
                            if activation_times[predecessor] < t and activation_times[predecessor] > -1 and (predecessor, node) not in disabled_edges: #these are the ones activated before
                                activeNeighborCounter += 1
                        if activeNeighborCounter >= threshold_dict[node]:
                            if node not in disabled_nodes:
                                activation_times[node] = t
                                newActivation = True
                                list_of_nodes_that_became_active.add(node)

                        if activeNeighborCounter > 0 and activation_times[node] == -1:
                            if random.random() <= probability_noisy:
                                if node not in disabled_nodes:
                                    activation_times[node] = t
                                    newActivation = True
                                    list_of_nodes_that_became_active.add(node)
        elif model == 'NOISY_SINGLE':
             assert probability_noisy is not None
             transmissions_tried = set()
             while newActivation and t < close_distance_approximation:
                t += 1
                newActivation = False

                for node in activation_times.keys():
                    if activation_times[node] == -1:
                        activeNeighborCounter = 0
                        activeNeighbors = set()
                        for predecessor in adjacency_dict[node]:
                            if activation_times[predecessor] < t and activation_times[predecessor] > -1 and (predecessor, node) not in disabled_edges: #these are the ones activated before
                                activeNeighborCounter += 1
                                activeNeighbors.add(predecessor)
                        if activeNeighborCounter >= threshold_dict[node]:
                            if node not in disabled_nodes:
                                activation_times[node] = t
                                newActivation = True
                                list_of_nodes_that_became_active.add(node)

                        if activeNeighborCounter > 0 and activation_times[node] == -1:
                            if random.random() <= probability_noisy and node not in transmissions_tried:
                                if node not in disabled_nodes:
                                    activation_times[node] = t
                                    newActivation = True
                                    list_of_nodes_that_became_active.add(node)
                            transmissions_tried.add(node)
        elif model == 'LTM':
            assert edgeWeights is not None

            while newActivation and t < close_distance_approximation:
                t += 1
                newActivation = False
                for node in activation_times.keys():
                    if activation_times[node] == -1:
                        activeNeighborCounter = 0
                        for predecessor in adjacency_dict[node]:
                            if activation_times[predecessor] < t and activation_times[predecessor] > -1 and (predecessor, node) not in disabled_edges: #these are the ones activated before
                                activeNeighborCounter += edgeWeights[(predecessor, node)] #this way it gets immediately weighted, otherwise its basically the GI model
                        if activeNeighborCounter >= threshold_dict[node]:
                            if node not in disabled_nodes:
                                activation_times[node] = t
                                newActivation = True
                                list_of_nodes_that_became_active.add(node)
        elif model == 'ICM': #todo: active_neibs_at_activation
            assert icm_probabilities is not None

            usedEdges = set()
            while newActivation and t < close_distance_approximation:
                t += 1
                newActivation = False
                #noisy first because it doesnt matter
                for node in activation_times.keys():
                    if activation_times[node] > -1:
                        #activeNeighborCounter = 0
                        for neib in adjacency_dict[node]:
                            if neib not in disabled_nodes and activation_times[neib] == -1 and random.random() <= icm_probabilities[(node,neib)] and (node,neib) not in usedEdges:
                                activation_times[neib] = t
                                newActivation = True
                                list_of_nodes_that_became_active.add(neib)
                            usedEdges.add((node,neib))
        else:
            raise NotImplementedError("Unknown Model")
        
        number_of_steps.append(t) 
        
        #count the number of nodes that became active
        cur_active_nodes = len(list_of_nodes_that_became_active)
        spreading_densities.append(cur_active_nodes)
    
    return np.mean(spreading_densities)/len(adjacency_dict.keys()), np.mean(number_of_steps)

def dropUntilDip(args):
    np.random.seed()

    adjacency_dict, edge_cpc, threshold_dict, number_of_seeds, disabled_nodes, disabled_edges, adjacency_dict_successors, seeding_function, dip_under_threshold, random_portion, random_factor = args

    cur_spread_Density,_ = calcSpreadingDensityForNetwork((adjacency_dict, threshold_dict, number_of_seeds, disabled_nodes, disabled_edges, adjacency_dict_successors, seeding_function, random_portion, random_factor))

    counter = 0

    while cur_spread_Density > dip_under_threshold:
        counter += 1

        #randomly disable an edge
        edge_to_disable = random.choice(list(edge_cpc.keys()))
        while edge_to_disable in disabled_edges:
            edge_to_disable = random.choice(list(edge_cpc.keys()))
        
        disabled_edges.add(edge_to_disable)
        node0, node1 = edge_to_disable

        disabled_edges.add((node1, node0))

        cur_spread_Density,_ = calcSpreadingDensityForNetwork((adjacency_dict, threshold_dict, number_of_seeds, disabled_nodes, disabled_edges, adjacency_dict_successors, seeding_function, random_portion, random_factor))

    return counter

def dropUntilDipNodes(args):
    np.random.seed()

    adjacency_dict, edge_cpc, threshold_dict, number_of_seeds, disabled_nodes, disabled_edges, adjacency_dict_successors, seeding_function, dip_under_threshold, random_portion, random_factor = args

    cur_spread_Density,_ = calcSpreadingDensityForNetwork((adjacency_dict, threshold_dict, number_of_seeds, disabled_nodes, disabled_edges, adjacency_dict_successors, seeding_function, random_portion, random_factor))

    counter = 0
    while cur_spread_Density > dip_under_threshold:
        counter += 1
        node_to_disable = random.choice(list(adjacency_dict.keys()))
        while node_to_disable in disabled_nodes:
            node_to_disable = random.choice(list(adjacency_dict.keys()))
        
        disabled_nodes.add(node_to_disable)

        cur_spread_Density,_ = calcSpreadingDensityForNetwork((adjacency_dict, threshold_dict, number_of_seeds, disabled_nodes, disabled_edges, adjacency_dict_successors, seeding_function, random_portion, random_factor))

    return counter

def plotGraph(G, layout=None):
    # Create a figure
    fig = plt.figure(figsize=(10, 10))

    if layout is None:
        pos = nx.circular_layout(G)
        # Adjust positions for every other node
        for j in range(0, len(G.nodes()), 2):
            pos[j] = pos[j] * 0.6
    else:
        pos = layout

    # Get the CPC values for nodes and scale them between 0 and 1 for alpha
    node_cpc = [G.nodes[node]["CPC"] for node in G.nodes]
    node_colors = [cpc / max(node_cpc) for cpc in node_cpc]  # Normalize to [0, 1]

    # Draw the network nodes with CPC values as the alpha for transparency
    nx.draw_networkx_nodes(G, pos, node_size=100, alpha=node_colors)

    # Get the CPC values for edges and scale them between 0 and 1 for alpha
    edge_cpc = [G.edges[edge]["CPC"] for edge in G.edges]
    edge_colors = [cpc / max(edge_cpc) for cpc in edge_cpc]  # Normalize to [0, 1]

    # Draw the network edges with CPC values as the alpha for transparency
    nx.draw_networkx_edges(G, pos, alpha=edge_colors, width=1, connectionstyle='arc3,rad=0.2', arrowsize=7)

    # Set the axes to be equal
    plt.axis('equal')

    # Show the plot
    plt.show()


class CpcHandler:
    def __init__(self, network, cores=4, seed_function=randomFactorSeed, sweeps=1, model='GI'):
        network = nx.DiGraph(network)
        self.network = network.copy()

        self.adjacency_dict = None
        self.adjacency_dict_successors = None
        self.thresholds = None
        self.cpc = None
        self.edge_cpc = None
        self.cores = cores

        #at least 5 runs per core otherwise they are useless
        if (self.network.number_of_nodes()*sweeps) < 5*cores:
            self.cores = int(self.network.number_of_nodes()*sweeps)//5
            self.cores = max(self.cores, 1)

        self.disabled_edges = set()
        self.disabled_nodes = set()

        self.seed_function = seed_function
        self.number_of_seeds = int(np.ceil(sweeps*self.network.number_of_nodes())) 
        self.snapshot = network.copy()
        self.random_portion = 0.02 #standard if not specified otherwise
        self.always_active_set = set()

        self.model = model
        self.edgeWeights = None
        self.probability_noisy = None
        self.icm_probabilities = None
    
    def to_dict_representation(self):
        #the aim is to convert this into a dict representation of a adjacency list
        self.adjacency_dict = {}
        for node in self.network.nodes:
            self.adjacency_dict[node] = list(self.network.predecessors(node))
        
        self.adjacency_dict_successors = {}
        for node in self.network.nodes:
            self.adjacency_dict_successors[node] = list(self.network.successors(node))
        
        self.edge_cpc = {}
        for edge in self.network.edges:
            self.edge_cpc[edge] = 0

    
    def setThresholds(self, T):
        self.thresholds = {}
        for node in self.adjacency_dict.keys():
            if isinstance(T, list):
                self.thresholds[node] = random.choice(T)
            elif T<1:
                self.thresholds[node] = np.ceil(len(self.adjacency_dict[node])*T)
            else:
                self.thresholds[node] = T

    def calcCPC(self, tqdm_bar = False):
        arguments = [(self.adjacency_dict, self.thresholds, math.ceil(self.number_of_seeds/self.cores), self.disabled_nodes, self.disabled_edges, self.adjacency_dict_successors, self.seed_function, tqdm_bar, self.always_active_set, self.random_portion, self.randomFactor, self.model, self.edgeWeights, self.probability_noisy, self.icm_probabilities)] * self.cores
        
        with Pool(self.cores) as pool:      
            results = pool.map(calcCPCForNetwork, arguments)

        causal_ambiguity = []

        self.cpc, self.edge_cpc,_ = results[0]

        for i in range(1, len(results)):
            for node in self.cpc.keys():
                self.cpc[node] += results[i][0][node]
            for edge in self.edge_cpc.keys():
                self.edge_cpc[edge] += results[i][1][edge]
        
        for i in range(len(results)):
            causal_ambiguity += results[i][2]
        
        if len(causal_ambiguity) > 0:
            self.causal_ambiguity = np.mean(causal_ambiguity)
        else:
            self.causal_ambiguity = 0
        
        if len(self.cpc.values()) == 0:
            maxNodeCPC = 1 #to prevent division by zero
        else:
            maxNodeCPC = max(self.cpc.values())
        
        if len(self.edge_cpc.values()) == 0:
            maxEdgeCPC = 1
        else:
            maxEdgeCPC = max(self.edge_cpc.values())

        if maxNodeCPC > 0:
            for node in self.cpc.keys():
                self.cpc[node] = self.cpc[node]/maxNodeCPC
        
        if maxEdgeCPC > 0:
            for edge in self.edge_cpc.keys():
                self.edge_cpc[edge] = self.edge_cpc[edge]/maxEdgeCPC
    
    def calcBetweenessForNetwork(self):
        betweenness = nx.edge_betweenness_centrality(self.network)
        for edge in self.network.edges:
            self.network.edges[edge]["betweenness"] = betweenness[edge]

    def getNetworkWithCPC(self):
        for node in self.network.nodes:
            self.network.nodes[node]["CPC"] = self.cpc[node]
            self.network.nodes[node]["T"] = self.thresholds[node]
        for edge in self.network.edges:
            self.network.edges[edge]["CPC"] = self.edge_cpc[edge]
            
        return self.network.copy()
    
    def getSpreadingDensity(self, with_steps = False):
        arguments = [(self.adjacency_dict, self.thresholds, math.ceil(self.number_of_seeds/self.cores), self.disabled_nodes, self.disabled_edges, self.adjacency_dict_successors, self.seed_function, self.random_portion, self.randomFactor, self.model, self.edgeWeights, self.probability_noisy, self.icm_probabilities)] * self.cores
        
        # Create a Pool of workers
        with Pool(self.cores) as pool:
            return_result = pool.map(calcSpreadingDensityForNetwork, arguments)
        
        results = [x[0] for x in return_result]
        avg_number_of_steps = np.mean([x[1] for x in return_result])


        self.spreadingDensity = np.mean(results)
        std_spreading_density = np.std(results)

        if with_steps:
            return self.spreadingDensity, avg_number_of_steps
        else:
            return self.spreadingDensity
    
    def calc_seeding_power(self, tqdm_bar = False):
        np.random.seed()

        adjacency_dict = self.adjacency_dict.copy()
        threshold_dict = self.thresholds.copy()
        number_of_seeds = self.number_of_seeds*10
        disabled_nodes = self.disabled_nodes.copy()
        disabled_edges = self.disabled_edges.copy()
        adjacency_dict_successors = self.adjacency_dict_successors.copy()
        seeding_function = self.seed_function

        spreading_densities = []

        activation_times = {}
        node_outcome_tracking = {}

        for node in adjacency_dict:
            activation_times[node] = -1
            node_outcome_tracking[node] = []

        isConverged = False

        i = 0
        progress_bar = tqdm(total=number_of_seeds)


        while not isConverged:
            i += 1
            progress_bar.update(1)
            if i > number_of_seeds:
                isConverged = True 

            #reset all activation times
            for node in activation_times:
                activation_times[node] = -1

            #1. create seed
            activation_times, list_of_nodes_that_became_active = seeding_function(adjacency_dict_successors, disabled_nodes, disabled_edges, activation_times)
            list_of_nodes_that_became_active = set(list_of_nodes_that_became_active)
            seed_nodes = list(list_of_nodes_that_became_active)

            #2. spread until convergence
            newActivation = True
            t = 0
            while newActivation:
                t += 1
                newActivation = False
                for node in activation_times.keys():
                    if activation_times[node] == -1:
                        activeNeighborCounter = 0
                        for predecessor in adjacency_dict[node]:
                            if activation_times[predecessor] < t and activation_times[predecessor] > -1 and (predecessor, node) not in disabled_edges: #these are the ones activated before
                                activeNeighborCounter += 1
                        if activeNeighborCounter >= threshold_dict[node] and node not in disabled_nodes:
                            activation_times[node] = t
                            newActivation = True
                            list_of_nodes_that_became_active.add(node)

            #count the number of nodes that became active
            cur_active_nodes = len(list_of_nodes_that_became_active)
            spreading_densities.append(cur_active_nodes/len(adjacency_dict.keys()))

            for node in seed_nodes:
                node_outcome_tracking[node].append(spreading_densities[-1])
        
        for node in node_outcome_tracking:
            node_outcome_tracking[node] = np.mean(node_outcome_tracking[node])
            self.network.nodes[node]['seeding_power'] = node_outcome_tracking[node]

    def makeSingleSpread(self, seed_nodes):
        adjacency_dict = self.adjacency_dict.copy()
        threshold_dict = self.thresholds.copy()
        disabled_nodes = self.disabled_nodes.copy()
        disabled_edges = self.disabled_edges.copy()

        activation_times = {}

        for node in adjacency_dict:
            activation_times[node] = -1

        #1. create seed
        for node in seed_nodes:
            activation_times[node] = 0

        list_of_nodes_that_became_active = set(seed_nodes)

        #2. spread until convergence
        newActivation = True
        t = 0
        while newActivation:
            t += 1
            newActivation = False
            for node in activation_times.keys():
                if activation_times[node] == -1:
                    activeNeighborCounter = 0
                    for predecessor in adjacency_dict[node]:
                        if activation_times[predecessor] < t and activation_times[predecessor] > -1 and (predecessor, node) not in disabled_edges: #these are the ones activated before
                            activeNeighborCounter += 1
                    if activeNeighborCounter >= threshold_dict[node] and node not in disabled_nodes:
                        activation_times[node] = t
                        newActivation = True
                        list_of_nodes_that_became_active.add(node)

        #count the number of nodes that became active
        cur_active_nodes = len(list_of_nodes_that_became_active)
        return cur_active_nodes/len(adjacency_dict.keys()), (list_of_nodes_that_became_active - set(seed_nodes))

    def calculateResilience(self, dip_under, trials, type = 'E'):
        # Define the list of arguments as tuples
        arguments = [(self.adjacency_dict, self.edge_cpc, self.thresholds, self.number_of_seeds, self.disabled_nodes, self.disabled_edges, self.adjacency_dict_successors, self.seed_function, dip_under, self.random_portion, self.randomFactor)] * trials
        
        # Create a Pool of workers
        with Pool(self.cores) as pool:
            if type == 'E':
                results = pool.map(dropUntilDip, arguments)
            else:
                results = pool.map(dropUntilDipNodes, arguments)
    
        return np.mean(results)

    def calc_divergence(self, tqdm_bar = False):
        if tqdm_bar:
            progress_bar = tqdm(total=len(self.network.nodes))

        for node in self.network.nodes:
            incoming, outgoing = 0, 0

            for neib in self.network.predecessors(node):
                incoming += self.edge_cpc[(neib, node)]
            for neib in self.network.successors(node):
                outgoing += self.edge_cpc[(node, neib)]
            
            self.network.nodes[node]['inflow'] = incoming
            self.network.nodes[node]['outflow'] = outgoing

            if incoming > 0:
                self.network.nodes[node]['divergence'] = outgoing/incoming
            else:
                self.network.nodes[node]['divergence'] = 0

            if tqdm_bar:
                progress_bar.update(1)
        
    def calc_symmetry(self):
        pairs = []
        for edge in self.edge_cpc.keys():
            pairs.append((self.edge_cpc[(edge[0], edge[1])], self.edge_cpc[(edge[1], edge[0])]))
        
        #create a dataframe
        df = pd.DataFrame(pairs, columns=['forward', 'backward'])

        symmetry = df.corr()['forward']['backward']

        return symmetry
    
    def calc_symmetry_cosine(self):
        pairs = []
        for edge in self.edge_cpc.keys():
            #print(edge)
            pairs.append((self.edge_cpc[(edge[0], edge[1])], self.edge_cpc[(edge[1], edge[0])]))
        
        #create a dataframe
        df = pd.DataFrame(pairs, columns=['forward', 'backward'])

        # Convert columns to vectors
        vec1 = df['forward'].values.reshape(1, -1)  # Reshape to 2D
        vec2 = df['backward'].values.reshape(1, -1)  # Reshape to 2D

        # Calculate cosine similarity
        similarity = cosine_similarity(vec1, vec2)

        return similarity[0][0]

    def calc_symmetry_delta(self):
        pairs = []
        for edge in self.edge_cpc.keys():
            delta = self.edge_cpc[(edge[0], edge[1])] - self.edge_cpc[(edge[1], edge[0])]
            delta = abs(delta)
            self.network.edges[edge]['delta'] = delta

        for node in self.cpc.keys():
            neibs_summed = list(self.network.successors(node)) + list(self.network.predecessors(node))
            delta = 0

            for neib in neibs_summed:
                if (node, neib) in self.network.edges:
                    delta += self.network.edges[(node, neib)]['delta']
                if (neib, node) in self.network.edges:
                    delta += self.network.edges[(neib, node)]['delta']

            self.network.nodes[node]['delta'] = delta
    

    def getNodeWithHighestCPC(self, doNotConsider = set()):
        activeSet = set(self.cpc.keys()) - self.disabled_nodes - doNotConsider

        if len(activeSet)==0:
            return random.choice(list(self.network.nodes()))
        
        maxOfActiveSet = max([self.cpc[node] for node in activeSet])   

        for node in activeSet:
            if self.cpc[node] == maxOfActiveSet:
                return node
    
    def getNodeWithLowestCPC(self, doNotConsider = set()):
        activeSet = set(self.cpc.keys()) - self.disabled_nodes - doNotConsider

        if len(activeSet)==0:
            return random.choice(list(self.network.nodes()))
        
        maxOfActiveSet = min([self.cpc[node] for node in activeSet])   

        for node in activeSet:
            if self.cpc[node] == maxOfActiveSet:
                return node

    def getEdgeWithHighestCPC(self):
        activeSet = set(self.edge_cpc.keys()) - self.disabled_edges
        maxOfActiveSet = max([self.edge_cpc[edge] for edge in activeSet])   

        for edge in activeSet:
            if self.edge_cpc[edge] == maxOfActiveSet:
                return edge
    
    def getEdgeWithLowestCPC(self):
        activeSet = set(self.edge_cpc.keys()) - self.disabled_edges
        maxOfActiveSet = min([self.edge_cpc[edge] for edge in activeSet])   

        for edge in activeSet:
            if self.edge_cpc[edge] == maxOfActiveSet:
                return edge
    
    def getEdgeWithHighestBetweeness(self):
        activeSet = set(self.network.edges) - self.disabled_edges
        maxOfActiveSet = max([self.network.edges[edge]['betweenness'] for edge in activeSet])   

        for edge in activeSet:
            if self.network.edges[edge]['betweenness'] == maxOfActiveSet:
                return edge
            
    def getEdgeWithLowestBetweenness(self):
        activeSet = set(self.network.edges) - self.disabled_edges
        maxOfActiveSet = min([self.network.edges[edge]['betweenness'] for edge in activeSet])   

        for edge in activeSet:
            if self.network.edges[edge]['betweenness'] == maxOfActiveSet:
                return edge
    
    def getNodeWithHighestDegree(self):
        max_degree_node = max(self.network.degree, key=lambda x: x[1])[0]
        return max_degree_node
    
    def getNodeWithHighestBetweeness(self):
        betweenness = nx.betweenness_centrality(self.network)
        max_betweenness_node = max(betweenness, key=betweenness.get)
        return max_betweenness_node

    def disableEdge(self, edge):
        self.disabled_edges.add(edge)
    
    def disableNode(self, node):
        self.disabled_nodes.add(node)

    def enableEdge(self, edge):
        self.disabled_edges.remove(edge)
    
    def enableNode(self, node):
        self.disabled_nodes.remove(node)
    
    def checkIfNodeIsDisabled(self, node):
        return node in self.disabled_nodes

    def getCPCofNode(self,node):
        return self.cpc[node]
    
    def addEdge(self, edge):
        #check if the edge is already in the network
        if edge not in self.network.edges:
            self.network.add_edge(*edge)
            self.to_dict_representation()
            return True
        
        return False
        
    def removeEdge(self, edge):
        self.network.remove_edge(*edge)
        self.to_dict_representation()

    def getTwoDisconnectedPairs(self):
        valid_choice = False

        while not valid_choice:
            edges = self.edge_cpc.keys()
            edge1 = random.choice(list(edges))
            edge2 = random.choice(list(edges))

            while random.random() > (1-self.edge_cpc[edge1]): #this is to increase the probability of picking low cpc edges
                edge1 = random.choice(list(edges))
            
            while random.random() < (1-self.edge_cpc[edge2]):
                edge2 = random.choice(list(edges))

            if edge1[0] != edge2[0] and edge1[1] != edge2[1]:
                newEdge1 = (edge1[0], edge2[1])
                newEdge2 = (edge2[0], edge1[1])
                if newEdge1 not in edges and newEdge2 not in edges:
                    valid_choice = True
        
        return edge1, edge2
    
    def replaceEdge(self, old_edge, new_edge):
        self.removeEdge(old_edge)
        self.addEdge(new_edge)
    
    def randomAddEdge(self, doNotCreateList=[]):
        valid_choice = False

        while not valid_choice:
            #randomly pick two nodes
            node1 = random.choice(list(self.network.nodes))
            node2 = random.choice(list(self.network.nodes))

            if node1 != node2 and not self.network.has_edge(node1, node2) and (node1, node2) not in doNotCreateList:
                valid_choice = True
        
        success = self.addEdge((node1, node2))
        assert success == True
    
    def makeSnapshot(self):
        self.snapshot = self.network.copy()
    
    def restoreSnapshot(self):
        self.network = self.snapshot.copy()
        self.to_dict_representation()
    
    def getTopNodes(self, portion = 0.05):
        sorted_nodes = sorted(self.cpc.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:int(len(self.network.nodes())*portion)]
    
    def setPortion(self, portion):
        self.random_portion = portion
    
    def setRandomFactor(self, randomFactor):
        self.randomFactor = randomFactor
    
    def setEdgeWeights(self, edgeWeight):
        self.edgeWeights = {}

        for edge in self.network.edges:
            self.edgeWeights[edge] = edgeWeight

    def setEdgeWeightsGaussian(self, mean, std):
        self.edgeWeights = {}

        for edge in self.network.edges:
            self.edgeWeights[edge] = np.random.normal(mean, std)
            
            if self.edgeWeights[edge] < 0:
                self.edgeWeights[edge] = 0
    
    def setProbabilityNoisy(self, probability):
        self.probability_noisy = probability
    
    def setProbabilitiesICM(self, probability):
        self.icm_probabilities = {}

        for edge in self.network.edges:
            self.icm_probabilities[edge] = probability
        