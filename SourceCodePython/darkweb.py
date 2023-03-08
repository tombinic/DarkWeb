import networkx as nx
import csv
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
from networkx.algorithms import community
import powerlaw
import math

def createGraph():
    with open('C:/Users/nicol/Documents/GitHub/DarkWeb/Dataset/darkweb.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        edges = []
        next(reader)
        for row in reader:
            edges.append((row[0], row[1], float(row[3])))

    graph = nx.DiGraph()
    graph.add_weighted_edges_from(edges)
    return graph

def authAndHubs(graph):
    hubs, authorities = nx.hits(graph, normalized=True, tol=1e-08)

    with open('C:/Users/nicol/Desktop/hubs1.csv', 'w', newline='') as csvfile:
        fieldnames = ['Id', 'Hubs', 'Auth']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for node, hubs in sorted(hubs.items(), key=lambda x: x[1], reverse=True):
            writer.writerow({'Id': node, 'Hubs': hubs, 'Auth': authorities[node]})

def inoutCloseness(graph):
    in_closeness = nx.closeness_centrality(graph)
    out_closeness = nx.closeness_centrality(graph.reverse())

    with open('C:/Users/nicol/Desktop/closeness.csv', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Node ID', 'In-Closeness Centrality', 'Out-Closeness Centrality'])
        for node_id in graph.nodes():
            writer.writerow([node_id, in_closeness[node_id], out_closeness[node_id]])

def initialStats(graph):
    n = graph.number_of_nodes()
    efficiency = 0.0
    for node in graph:
        sp = nx.shortest_path_length(graph, source=node)
        efficiency += sum(1.0/l for l in sp.values() if l > 0)
    efficiency /= (n*(n-1))
    print("Efficiency ", efficiency)
    print("Density " + str(nx.density(graph)))
    print("Assortativity coefficient " + str(nx.degree_assortativity_coefficient(graph, weight="weight")))
    print("Number of isolated nodes " + str(nx.number_of_isolates(graph)))
    print("Number of self loops " + str(nx.number_of_selfloops(graph)))
    scc = max(nx.strongly_connected_components(graph), key=len)
    print("Dimension of the largest strongly connected component: " + str(len(scc)))
    wcc = max(nx.weakly_connected_components(graph), key=len)
    print("Dimension of the largest weakly connected component: " + str(len(wcc)))
    n_scc = nx.number_strongly_connected_components(graph)
    print("Number of all the strongly connected components: " + str(n_scc))
    n_wcc = nx.number_weakly_connected_components(graph)
    print("Number of all the weakly connected components: " + str(n_wcc))
    scc_subgraphs = (graph.subgraph(c) for c in nx.strongly_connected_components(graph))
    largest_scc = max(scc_subgraphs, key=len)
    in_component = set()
    out_component = set()

    for node in largest_scc.nodes():
        predecessors = graph.predecessors(node)
        in_component.update(pred for pred in predecessors if pred not in largest_scc)

    for node in largest_scc.nodes():
        successors = graph.successors(node)
        out_component.update(succ for succ in successors if succ not in largest_scc)


    out_component_new = out_component.copy()
    while(True):
        out_component_new = out_component.copy()

        for node in out_component_new:
            successors = graph.successors(node)
            out_component.update(succ for succ in successors if succ not in largest_scc)

        if out_component == out_component_new:
            break

    print("Dimension of the largest IN component (IN): " + str(len(in_component)))
    print("Dimension of the largest OUT component (OUT): " + str(len(out_component)))
    avg_path_scc = nx.average_shortest_path_length(largest_scc)
    print("Average shortest path length in SCC: " + str(avg_path_scc))      

def inDegreeDistribution(graph):
    degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)
    degreeCount = Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cnt = tuple(i/graph.number_of_nodes() for i in cnt)
    plt.title("In-Degree distribution")
    plt.xlabel("k")
    plt.ylabel("Pin(k)")
    plt.plot(deg, cnt, color='red', linestyle='solid')
    plt.show()
    plt.title("In-Degree cumulated distribution")
    cs = np.cumsum(cnt)
    plt.xlabel("k")
    plt.ylabel(r'$\bar{P}$' + "(k)",fontsize=12)
    plt.plot(deg, cs, color='blue', linestyle='solid')
    plt.show()

def outDegreeDistribution(graph):
    degree_sequence = sorted([d for n, d in graph.out_degree()], reverse=True)
    degreeCount = Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cnt = tuple(i/graph.number_of_nodes() for i in cnt)
    plt.title("Out-Degree distribution")
    plt.xlabel("k")
    plt.ylabel("Pout(k)")
    plt.plot(deg, cnt, color='red', linestyle='solid')
    plt.show()
    plt.title("Out-Degree cumulated distribution")
    cs = np.cumsum(cnt)
    plt.xlabel("k")
    plt.ylabel(r'$\bar{P}$' + "(k)",fontsize=12)
    plt.plot(deg, cs, color='blue', linestyle='solid')
    plt.show()

def inStrengthDistribution(graph):
    degree_sequence = sorted([d for n, d in graph.in_degree(weight="weight")], reverse=True)
    degreeCount = Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cnt = tuple(i/graph.number_of_nodes() for i in cnt)
    plt.title("In-Strength distribution")
    plt.xlabel("s")
    plt.ylabel("Pin(s)")
    plt.plot(deg, cnt, color="red")
    plt.show()
    plt.title("In-Strength cumulated distribution")
    cs = np.cumsum(cnt)
    plt.xlabel("s")
    plt.ylabel(r'$\bar{P}$' + "(s)",fontsize=12)
    plt.plot(deg, cs)
    plt.show()

def outStrengthDistribution(graph):
    degree_sequence = sorted([d for n, d in graph.out_degree(weight="weight")], reverse=True)
    degreeCount = Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cnt = tuple(i/graph.number_of_nodes() for i in cnt)
    plt.title("Out-Strength distribution")
    plt.xlabel("s")
    plt.ylabel("Pout(s)")
    plt.plot(deg, cnt, color="red")
    plt.show()
    plt.title("Out-Strength cumulated distribution")
    cs = np.cumsum(cnt)
    plt.xlabel("s")
    plt.ylabel(r'$\bar{P}$' + "(s)",fontsize=12)
    plt.plot(deg, cs)
    plt.show()

def nodesInDegreePercentage(graph):
    degree_sequence = sorted([d for n, d in graph.in_degree()], reverse=True)
    degreeCount = Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cnt = tuple((i/graph.number_of_nodes()) * 100 for i in cnt)
    plt.plot(deg, cnt, color="purple")
    plt.xlim([0, 20])
    plt.ylim([0.01, 100])
    plt.title('In-Degree and nodes percentage')
    plt.xlabel('In-Degree')
    plt.ylabel('Nodes percentage')
    plt.show()
    
def nodesInStrengthPercentage(graph):
    degree_sequence = sorted([d for n, d in graph.in_degree(weight="weight")], reverse=True)
    degreeCount = Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cnt = tuple((i/graph.number_of_nodes()) * 100 for i in cnt)
    plt.plot(deg, cnt, color="purple")
    plt.xlim([0, 20])
    plt.ylim([0.01, 100])
    plt.title('In-Strength and nodes percentage')
    plt.xlabel('In-Strength')
    plt.ylabel('Nodes percentage')
    plt.show()

def nodesOutStrengthPercentage(graph):
    degree_sequence = sorted([d for n, d in graph.out_degree(weight="weight")], reverse=True)
    degreeCount = Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cnt = tuple((i/graph.number_of_nodes()) * 100 for i in cnt)
    plt.plot(deg, cnt, color="purple")
    plt.xlim([0, 20])
    plt.ylim([0.01, 100])
    plt.title('Out-Strength and nodes percentage')
    plt.xlabel('Out-Strength')
    plt.ylabel('Nodes percentage')
    plt.show()

def nodesOutDegreePercentage(graph):
    degree_sequence = sorted([d for n, d in graph.out_degree()], reverse=True)
    degreeCount = Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cnt = tuple((i/graph.number_of_nodes()) * 100 for i in cnt)
    plt.plot(deg, cnt, color="purple")
    #plt.yscale("log")
    plt.xlim([0, 20])
    plt.ylim([0.01, 100])
    plt.title('Out-Degree and nodes percentage')
    plt.xlabel('Out-Degree')
    plt.ylabel('Nodes percentage')
    plt.show()

def nodesPercentageIncomingLink(graph):
    nodes_with_one_in_link = [n for n in graph.nodes() if graph.in_degree(n) == 1]
    outdegree = dict(graph.out_degree())
    top_nodes = sorted(outdegree, key=outdegree.get, reverse=True)[:5]

    selected_edges = [(u, v) for (u, v) in graph.edges() if u in top_nodes]
    selected_graph = nx.DiGraph(selected_edges)
    nodes_with_one_incoming_edge_from_out_degree_hubs = [n for n in selected_graph.nodes() if selected_graph.in_degree(n) == 1]

    print("Percentage of nodes with one incoming edge from out degree hubs = " + str((len(nodes_with_one_incoming_edge_from_out_degree_hubs)*100)/(len(nodes_with_one_in_link))))

def shortestPathAnalysisScc(graph):
    scc_subgraphs = (graph.subgraph(c) for c in nx.strongly_connected_components(graph))
    largest_scc = max(scc_subgraphs, key=len)
    shortest_paths = nx.shortest_path(largest_scc)
    path_lengths = {}
    total_paths = 0
    for source in shortest_paths:
        for target in shortest_paths[source]:
            path_length = len(shortest_paths[source][target]) - 1
            total_paths += 1
            if path_length not in path_lengths:
                path_lengths[path_length] = 1
            else:
                path_lengths[path_length] += 1

    path_percentages = {}
    for length, count in path_lengths.items():
        path_percentages[length] = count / total_paths * 100

    plt.bar(list(path_percentages.keys()), list(path_percentages.values()))
    plt.title("Average Shortest Path SCC")
    plt.xlabel('Shortest Path Length SCC')
    plt.ylabel('Nodes percentage')
    plt.show()

def shortestPathAnalysisWcc(graph):
    shortest_paths = nx.shortest_path(graph)
    path_lengths = {}
    total_paths = 0
    for source in shortest_paths:
        for target in shortest_paths[source]:
            path_length = len(shortest_paths[source][target]) - 1
            total_paths += 1
            if path_length not in path_lengths:
                path_lengths[path_length] = 1
            else:
                path_lengths[path_length] += 1

    path_percentages = {}
    for length, count in path_lengths.items():
        path_percentages[length] = count / total_paths * 100

    plt.bar(list(path_percentages.keys()), list(path_percentages.values()))
    plt.title("Average Shortest Path WCC")
    plt.xlabel('Shortest Path Length WCC')
    plt.ylabel('Nodes percentage')
    plt.show()

def robustnessAttackLargestWcc(graph):
    initial_size_wcc = len(max(nx.weakly_connected_components(graph), key=len))
    sizes_random = []
    removed_perc_random = []
    sizes_pr = []
    removed_perc_pr = []
    sizes_hub = []
    removed_perc_hub = []
    sizes_out_degree = []
    removed_perc_out_degree = []
    G_attack_wcc = graph.copy()

    pr_wcc = nx.pagerank(G_attack_wcc, weight='weight')
    hubs_wcc = nx.hits(G_attack_wcc, normalized=True, tol=1e-08)[0]

    #print(out_degree_wcc)
    sorted_nodes_pr = sorted(pr_wcc.items(), key=lambda x: x[1], reverse=True)
    sorted_nodes_hubs = sorted(hubs_wcc.keys(), key=lambda x: hubs_wcc[x], reverse=True)
    sorted_nodes_out_degree = sorted(graph.out_degree(), key=lambda x: x[1], reverse=True)
    #print(top_nodes_out_degree)
    top_nodes_pr = [x[0] for x in sorted_nodes_pr]
    top_nodes_hubs = [node for node in sorted_nodes_hubs]
    top_nodes_out_degree = [x[0] for x in sorted_nodes_out_degree]

    for i in range(0, (initial_size_wcc // 2)):      
        node_to_remove = np.random.choice(G_attack_wcc.nodes, size=1, replace=False)  
        G_attack_wcc.remove_nodes_from(node_to_remove)
        #print(len(G_attack_wcc.nodes))
        size = len(max(nx.weakly_connected_components(G_attack_wcc), key=len))
        #print(str(size) + " " + str(nx.number_weakly_connected_components(G_attack_wcc)))
        sizes_random.append(size / initial_size_wcc * 100)
        removed_perc_random.append(i / initial_size_wcc * 100)
    
    G_attack_wcc = graph.copy()
    for i in range(0, (initial_size_wcc // 2)):        
        G_attack_wcc.remove_node(top_nodes_pr[i])
        size = len(max(nx.weakly_connected_components(G_attack_wcc), key=len))
        #print(str(size) + " " + str(nx.number_weakly_connected_components(G_attack_wcc)))
        sizes_pr.append(size / initial_size_wcc * 100)
        removed_perc_pr.append(i / initial_size_wcc * 100)

    G_attack_wcc = graph.copy()
    for i in range(0, (initial_size_wcc // 2)):        
        G_attack_wcc.remove_node(top_nodes_hubs[i])
        size = len(max(nx.weakly_connected_components(G_attack_wcc), key=len))
        #print(str(size) + " " + str(nx.number_weakly_connected_components(G_attack_wcc)))
        sizes_hub.append(size / initial_size_wcc * 100)
        removed_perc_hub.append(i / initial_size_wcc * 100)
    
    G_attack_wcc = graph.copy()
    for i in range(0, (initial_size_wcc // 2)):        
        G_attack_wcc.remove_node(top_nodes_out_degree[i])
        size = len(max(nx.weakly_connected_components(G_attack_wcc), key=len))
        #print(str(size) + " " + str(nx.number_weakly_connected_components(G_attack_wcc)))
        sizes_out_degree.append(size / initial_size_wcc * 100)
        removed_perc_out_degree.append(i / initial_size_wcc * 100)

    plt.plot(removed_perc_random, sizes_random, linestyle='-', color='blue', label="Random")
    plt.plot(removed_perc_pr, sizes_pr, linestyle='-', color='red', label="PageRank")
    plt.plot(removed_perc_hub, sizes_hub, linestyle='-', color='orange', label="Hubs")
    plt.plot(removed_perc_out_degree, sizes_out_degree, linestyle='-', color='yellow', label="Out-Degree")
    plt.xlabel('Nodes percentage removed')
    plt.ylabel('Largest WCC dimension')
    plt.legend()
    plt.show()

def robustnessAttackEfficiency(graph):
    initial_size_wcc = len(max(nx.weakly_connected_components(graph), key=len))
    efficiency_random = []
    removed_perc_random = []
    efficiency_pr = []
    removed_perc_pr = []
    G_attack_wcc = graph.copy()

    pr_wcc = nx.pagerank(G_attack_wcc, weight='weight')
    sorted_nodes_pr = sorted(pr_wcc.items(), key=lambda x: x[1], reverse=True)
    top_nodes_pr = [x[0] for x in sorted_nodes_pr]

    for i in range(0, 600):      
        node_to_remove = np.random.choice(G_attack_wcc.nodes, size=1, replace=False)  
        G_attack_wcc.remove_nodes_from(node_to_remove)

        n = graph.number_of_nodes()
        efficiency = 0.0
        for node in G_attack_wcc:
            sp = nx.shortest_path_length(G_attack_wcc, source=node)
            efficiency += sum(1.0/l for l in sp.values() if l > 0)
        efficiency /= (n*(n-1))

        efficiency_random.append(efficiency)
        removed_perc_random.append(i / initial_size_wcc * 100)
    
    G_attack_wcc = graph.copy()
    for i in range(0, 600):        
        G_attack_wcc.remove_node(top_nodes_pr[i])

        n = graph.number_of_nodes()

        efficiency = 0.0
        for node in G_attack_wcc:
            sp = nx.shortest_path_length(G_attack_wcc, source=node)
            efficiency += sum(1.0/l for l in sp.values() if l > 0)
        efficiency /= (n*(n-1))

        efficiency_pr.append(efficiency)
        removed_perc_pr.append(i / initial_size_wcc * 100)
    
    plt.plot(removed_perc_random, efficiency_random, linestyle='-', color='blue', label="Random")
    plt.plot(removed_perc_pr, efficiency_pr, linestyle='-', color='red', label="PageRank")
    plt.xlabel('Nodes percentage removed')
    plt.ylabel('WCC efficiency')
    plt.legend()
    plt.show()

def communityAnalysisGreedy(graph):
    communities = nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")
    partition_sizes = {i: len(c) / len(graph.nodes) * 100 for i, c in enumerate(communities)}
    print(partition_sizes)
    plt.bar(range(len(partition_sizes)), list(partition_sizes.values()), align='center')
    plt.xticks(range(len(partition_sizes)), list(partition_sizes.keys()))
    plt.xlabel('Partition')
    plt.ylabel('Nodes percentage per communities')
    plt.show()

def communityAnalysisLouvain(graph):
    louvain_communities = community.louvain_communities(graph, weight="weight")
    partition_sizes = {i: len(c) / len(graph.nodes) * 100 for i, c in enumerate(louvain_communities)}
    plt.bar(range(len(partition_sizes)), list(partition_sizes.values()), align='center')
    plt.xticks(range(len(partition_sizes)), list(partition_sizes.keys()))
    plt.xlabel('Partition')
    plt.ylabel('Nodes percentage per communities')
    plt.show()

def communityAnalysisGirvanNewman(graph):
    gn_communities = community.girvan_newman(graph)
    partition_tmp = [sorted(c) for c in next(gn_communities)]
    print(partition_tmp)
    '''
    partition_sizes = {i: len(c) / len(graph.nodes) * 100 for i, c in enumerate(louvain_communities)}
    plt.bar(range(len(partition_sizes)), list(partition_sizes.values()), align='center')
    plt.xticks(range(len(partition_sizes)), list(partition_sizes.keys()))
    plt.xlabel('Partition')
    plt.ylabel('Nodes percentage per communities')
    plt.show()
    '''

def betweennessCentrality(graph):
    greens = np.linspace(1, 0, 5)
    colors = [(0, green, 0) for green in greens]
    centrality = nx.betweenness_centrality(graph)
    top_nodes = sorted(centrality.keys(), key=lambda x: centrality[x], reverse=True)[:5]

    x = [node[:8] + '...' for node in top_nodes]
    y = [centrality[node] for node in top_nodes]

    plt.bar(x, y, color=colors)
    plt.xlabel('Node')
    plt.ylabel('Betweenness Centrality')
    plt.show()

def inClosenessCentrality(graph):
    yellows = np.linspace(1, 0, 5)
    colors = [(yellow, yellow, 0) for yellow in yellows]
    centrality = nx.closeness_centrality(graph)
    top_nodes = sorted(centrality.keys(), key=lambda x: centrality[x], reverse=True)[:5]
   
    x = [node[:8] + '...' for node in top_nodes]
    y = [centrality[node] for node in top_nodes]

    plt.bar(x, y, color=colors)
    plt.xlabel('Node')
    plt.ylabel('In-Closeness Centrality')
    plt.show()

def outClosenessCentrality(graph):
    pinks = np.linspace(1, 0, 5)
    colors = [(pink, 0.5*pink, 0.5*pink) for pink in pinks]
    centrality = nx.closeness_centrality(graph.reverse())
    top_nodes = sorted(centrality.keys(), key=lambda x: centrality[x], reverse=True)[:5]
   
    x = [node[:8] + '...' for node in top_nodes]
    y = [centrality[node] for node in top_nodes]

    plt.bar(x, y, color=colors)
    plt.xlabel('Node')
    plt.ylabel('Out-Closeness Centrality')
    plt.show()

def pageRankCentrality(graph):
    blues = np.linspace(1, 0, 5)
    colors = [(0, 0, blue) for blue in blues]
    centrality = nx.pagerank(graph, weight="weight")
    top_nodes = sorted(centrality.keys(), key=lambda x: centrality[x], reverse=True)[:5]
    
    x = [node[:8] + '...' for node in top_nodes]
    y = [centrality[node] for node in top_nodes]

    plt.bar(x, y, color=colors)
    plt.xlabel('Node')
    plt.ylabel('PageRank Centrality')
    plt.show()

def hubScore(graph):
    mixed = np.linspace(0, 1, 5)
    colors = [(0, green, blue) for green, blue in zip(mixed, reversed(mixed))]
    centrality = nx.hits(graph, normalized=True, tol=1e-08)[0]

    top_nodes = sorted(centrality.keys(), key=lambda x: centrality[x], reverse=True)[:5]

    x = [node[:8] + '...' for node in top_nodes]
    y = [centrality[node] for node in top_nodes]

    plt.bar(x, y, color=colors)
    plt.yscale("log")
    plt.xlabel('Node')
    plt.ylabel('Hub Score')
    plt.show()

def authScore(graph):
    oranges = np.linspace(1, 0, 5)
    colors = [(orange, orange/2, 0) for orange in oranges]
    centrality = nx.hits(graph, normalized=True, tol=1e-08)[1]

    top_nodes = sorted(centrality.keys(), key=lambda x: centrality[x], reverse=True)[:5]

    x = [node[:8] + '...' for node in top_nodes]
    y = [centrality[node] for node in top_nodes]

    plt.bar(x, y, color=colors)
    plt.yscale("log")
    plt.xlabel('Node')
    plt.ylabel('Authority Score')
    plt.show()

def inDegreeCentrality(graph):
    reds = np.linspace(1, 0, 5) 
    colors = [(red, 0, 0) for red in reds] 
    centrality = dict(graph.in_degree())
    top_nodes = sorted(centrality.keys(), key=lambda x: centrality[x], reverse=True)[:5]

    x = [node[:8] + '...' for node in top_nodes]
    y = [centrality[node] for node in top_nodes]

    plt.bar(x, y, color=colors)
    plt.xlabel('Nodes')
    plt.ylabel('In-Degree Score')
    plt.show()

def inStrengthCentrality(graph):
    n = 5
    colors = [(purple, 0, purple) for purple in np.linspace(1, 0, n)]
    centrality = dict(graph.in_degree(weight="weight"))
    top_nodes = sorted(centrality.keys(), key=lambda x: centrality[x], reverse=True)[:5]

    x = [node[:8] + '...' for node in top_nodes]
    y = [centrality[node] for node in top_nodes]

    plt.bar(x, y, color=colors)
    plt.xlabel('Node')
    plt.ylabel('In-Strength Score')
    plt.show()

def outDegreeCentrality(graph):
    n = 5
    colors = [(0, 0, blue) for blue in np.linspace(1, 0, n)]
    centrality = dict(graph.out_degree())
    top_nodes = sorted(centrality.keys(), key=lambda x: centrality[x], reverse=True)[:5]

    x = [node[:8] + '...' for node in top_nodes]
    y = [centrality[node] for node in top_nodes]

    plt.bar(x, y, color=colors)
    plt.xlabel('Node')
    plt.ylabel('Out-Degree Score')
    plt.show()

def outStrengthCentrality(graph):
    n = 5
    colors = [(0, gray, gray) for gray in np.linspace(1, 0, n)]
    centrality = dict(graph.out_degree(weight="weight"))
    top_nodes = sorted(centrality.keys(), key=lambda x: centrality[x], reverse=True)[:5]

    x = [node[:8] + '...' for node in top_nodes]
    y = [centrality[node] for node in top_nodes]

    plt.bar(x, y, color=colors)
    plt.xlabel('Node')
    plt.ylabel('Out-Strength Score')
    plt.show()

def linkPredictionJaccard(graph):
    undirected_graph = graph.to_undirected()
    similarity = list(nx.algorithms.link_prediction.jaccard_coefficient(undirected_graph))
    sorted_similarity = sorted(similarity, key=lambda x: x[2], reverse=True)

    x_labels = [f"({u}, {v})" for u, v, p in sorted_similarity[:5]]
    y_values = [p for u, v, p in sorted_similarity[:5]]

    plt.bar(x_labels, y_values)
    plt.ylabel("Similarity")
    plt.title("Top 5 link prediction results")
    plt.show()

def linkPredictionPreferentialAttachment(graph):
    undirected_graph = graph.to_undirected()
    similarity = list(nx.algorithms.link_prediction.preferential_attachment(undirected_graph))
    sorted_similarity = sorted(similarity, key=lambda x: x[2], reverse=True)

    x_labels = [f"({u}, {v})" for u, v, p in sorted_similarity[:5]]
    y_values = [p for u, v, p in sorted_similarity[:5]]

    plt.bar(x_labels, y_values)
    plt.ylabel("Similarity")
    plt.title("Top 5 link prediction results")
    plt.show()

def linkPredictionResourceAllocationIndex(graph):
    undirected_graph = graph.to_undirected()
    similarity = list(nx.algorithms.link_prediction.resource_allocation_index(undirected_graph))
    sorted_similarity = sorted(similarity, key=lambda x: x[2], reverse=True)

    x_labels = [f"({u}, {v})" for u, v, p in sorted_similarity[:5]]
    y_values = [p for u, v, p in sorted_similarity[:5]]

    plt.bar(x_labels, y_values)
    plt.ylabel("Similarity")
    plt.title("Top 5 link prediction results")
    plt.show()

def linkPredictionCommonNeighbours(graph):
    undirected_graph = graph.to_undirected()
    similarity = list(nx.algorithms.link_prediction.common_neighbor_centrality(undirected_graph))  
    sorted_similarity = sorted(similarity, key=lambda x: x[2], reverse=True)

    x_labels = [f"({u}, {v})" for u, v, p in sorted_similarity[:5]]
    y_values = [p for u, v, p in sorted_similarity[:5]]

    plt.bar(x_labels, y_values)
    plt.ylabel("Similarity")
    plt.title("Top 5 link prediction results")
    plt.show()

def inverse_community_mapping(partition):
    partition_mapping = {}
    internal_degrees = {}
    for c in range(min(partition.values()),max(partition.values()) + 1):
        partition_mapping[c] = [k for k, v in partition.items() if v == c]
        internal_degrees[c] = 0

    return partition_mapping, internal_degrees

def kshell(graph):
    colors = [
    "red", "blue", "green", "purple", "orange", "brown", "black", "gray", "navy", 
    "teal", "magenta", "maroon", "olive", "pink", "gold", "silver", "cyan", 
    "indigo", "lime", "orchid", "peru", "plum", "salmon", "sienna", "skyblue", 
    "tan", "thistle", "turquoise"
    ]
    
    core_number = nx.algorithms.core.core_number(graph)
    ks, _ = inverse_community_mapping(core_number)
    print(ks.keys())
    color_map = {}
    for i, (key, value) in enumerate(ks.items()):
        color_map.update(dict.fromkeys(value, colors[i]))
    figure, axes = plt.subplots()
    r = 100
    points = {}
    for key, value in ks.items():
        delta = 2*math.pi/len(value)
        for i, j in enumerate(value):
            points[j] = (r * math.cos(i * delta), r * math.sin(i * delta))
        r -= 5
    nx.draw(graph, pos=points, with_labels=False, node_size=10, edge_color="#ccc", node_color=[color_map[node] for node in graph.nodes()])

    r = 100
    for key, value in ks.items():
        delta = 2*math.pi/len(value)
        for i, j in enumerate(value):
            points[j] = (r * math.cos(i * delta), r * math.sin(i * delta))
        circle = plt.Circle((0, 0), r, color="red", fill=False)
        axes.add_artist(circle)
        r -= 5

    axes.set_xlim((-100, 100))
    axes.set_ylim((-100, 100))
    axes.set_xticks([])
    axes.set_yticks([])
    plt.show()
   

def main():
    graph = createGraph()
    inoutCloseness(graph)
    #initialStats(graph)                -- done
    #inDegreeDistribution(graph)        -- done
    #outDegreeDistribution(graph)       -- done
    #inStrengthDistribution(graph)      -- done
    #outStrengthDistribution(graph)     -- done
    #nodesInDegreePercentage(graph)     -- done
    #nodesInStrengthPercentage(graph)   -- done
    #nodesOutDegreePercentage(graph)    -- done
    #nodesOutStrengthPercentage(graph)  -- done
    #nodesPercentageIncomingLink(graph) -- done
    #shortestPathAnalysisScc(graph)
    #shortestPathAnalysisWcc(graph)
    #robustnessAttackLargestWcc(graph)
    #robustnessAttackEfficiency(graph)
    #communityAnalysisGreedy(graph)
    #communityAnalysisGirvanNewman(graph)
    #betweennessCentrality(graph)       -- done
    #inClosenessCentrality(graph)       -- done
    #outClosenessCentrality(graph)      -- done
    #pageRankCentrality(graph)          -- done
    #hubScore(graph)                    -- done
    #authScore(graph)                   -- done
    #inDegreeCentrality(graph)             -- done
    #outDegreeCentrality(graph)            -- done
    #inStrengthCentrality(graph)           -- done
    #outStrengthCentrality(graph)          -- done
    #linkPredictionJaccard(graph)
    #linkPredictionPreferentialAttachment(graph)
    #linkPredictionResourceAllocationIndex(graph)
    #linkPredictionCommonNeighbours(graph)
    #kshell(graph)


main()