import networkx as nx
import csv
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter

'''

# robustness - random attack 
initial_size = len(G.nodes)
initial_size_scc = 297
# Inizializza le liste per i risultati del random attack
sizes = []
removed_perc = []

G_attack = G.copy()
scc_subgraphs = (G.subgraph(c) for c in nx.strongly_connected_components(G))
largest_scc = max(scc_subgraphs, key=len)
G_attack = largest_scc.copy()

pr = nx.pagerank(G_attack, weight='weight')

# crea una lista ordinata di coppie (nodo, pagerank)
pr_sorted = sorted(pr.items(), key=lambda x: x[1], reverse=True)
print(pr_sorted)
# crea una lista ordinata di nodi in ordine decrescente di PageRank centrality
top_nodes = [x[0] for x in pr_sorted]

print(top_nodes)
#vedere perche le pagerank sono diverse

for i in range(1, initial_size_scc // 2 + 1):
    
    # Sceglie casualmente un nodo da rimuovere
    #nodes_to_remove = np.random.choice(G_attack, replace=False)
    nodes_to_remove = top_nodes[i]
    # Rimuove il nodo
    
    G_attack.remove_node(nodes_to_remove)

    # Calcola la dimensione rimanente del grafo dopo il random attack
    #size = len(G_attack.nodes)
    size = len(max(nx.strongly_connected_components(G_attack), key=len))
    sizes.append(size / initial_size_scc * 100)
    removed_perc.append(i / initial_size_scc * 100)


print(size)
plt.plot(removed_perc, sizes, linestyle='-', color='blue')
plt.xlabel('Percentuale di nodi rimossi')
plt.ylabel('Dimensione del grafo in termini di nodi')
plt.show()
'''

def createGraph():
    with open('C:/Users/nicol/Desktop/darkweb.csv', 'r') as csvfile:
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

def initialStats(graph):
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
    degree_sequence = sorted([d for n, d in graph.in_degree()], reverse=True)
    degreeCount = Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cnt = tuple(i/graph.number_of_nodes() for i in cnt)
    plt.title("In-Degree distribution")
    plt.xlabel("k")
    plt.ylabel("Pin(k)")
    plt.plot(deg, cnt)
    plt.show()
    plt.title("In-Degree cumulated distribution")
    cs = np.cumsum(cnt)
    plt.xlabel("k")
    plt.ylabel(r'$\bar{P}$' + "(k)",fontsize=12)
    plt.plot(deg, cs)
    plt.show()

def outDegreeDistribution(graph):
    degree_sequence = sorted([d for n, d in graph.out_degree()], reverse=True)
    degreeCount = Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cnt = tuple(i/graph.number_of_nodes() for i in cnt)
    plt.title("Out-Degree distribution")
    plt.xlabel("k")
    plt.ylabel("Pout(k)")
    plt.plot(deg, cnt)
    plt.show()
    plt.title("Out-Degree cumulated distribution")
    cs = np.cumsum(cnt)
    plt.xlabel("k")
    plt.ylabel(r'$\bar{P}$' + "(k)",fontsize=12)
    plt.plot(deg, cs)
    plt.show()

def inStrengthDistribution(graph):
    degree_sequence = sorted([d for n, d in graph.in_degree(weight="weight")], reverse=True)
    degreeCount = Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cnt = tuple(i/graph.number_of_nodes() for i in cnt)
    plt.title("In-Strength distribution")
    plt.xlabel("s")
    plt.ylabel("Pin(s)")
    plt.plot(deg, cnt)
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
    plt.plot(deg, cnt)
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
    plt.plot(deg, cnt)
    plt.yscale("log")
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
    plt.plot(deg, cnt)
    plt.yscale("log")
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
    plt.plot(deg, cnt)
    plt.yscale("log")
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
    plt.plot(deg, cnt)
    plt.yscale("log")
    plt.xlim([0, 20])
    plt.ylim([0.01, 100])
    plt.title('Out-Degree and nodes percentage')
    plt.xlabel('Out-Degree')
    plt.ylabel('Nodes percentage')
    plt.show()

def nodesPercentageIncomingLink(graph):
# nodes percentage which have the in link coming from the most 5 out degree nodes
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


def main():
    graph = createGraph()
    initialStats(graph)
    #inDegreeDistribution(graph)
    #outDegreeDistribution(graph)
    #inStrengthDistribution(graph)
    #outStrengthDistribution(graph)
    #nodesInDegreePercentage(graph)
    #nodesInStrengthPercentage(graph)
    #nodesOutDegreePercentage(graph)
    #nodesOutStrengthPercentage(graph)
    #nodesPercentageIncomingLink(graph)
    #shortestPathAnalysisScc(graph)
    #shortestPathAnalysisWcc(graph)

main()