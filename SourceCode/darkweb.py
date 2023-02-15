import networkx as nx
import csv
import numpy as np
import matplotlib.pyplot as plt

with open('C:/Users/nicol/Desktop/darkweb.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=";")
    edges = []
    next(reader)
    for row in reader:
        #print(row[0])
        edges.append((row[0], row[1], float(row[3])))

# Crea un nuovo grafo vuoto
G = nx.DiGraph()

# Aggiungi gli archi al grafo con i rispettivi pesi
G.add_weighted_edges_from(edges)

# Visualizza il grafo
print(G)

# authority and hubs on weighted network

hubs, authorities = nx.hits(G, normalized=True, tol=1e-08)

with open('C:/Users/nicol/Desktop/hubs1.csv', 'w', newline='') as csvfile:
    fieldnames = ['Id', 'Hubs', 'Auth']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for node, hubs in sorted(hubs.items(), key=lambda x: x[1], reverse=True):
        writer.writerow({'Id': node, 'Hubs': hubs, 'Auth': authorities[node]})

# assortativity network
print(nx.degree_assortativity_coefficient(G, weight="weight"))

# number of isolated nodes
print(nx.number_of_isolates(G))

# number of self loops
print(nx.number_of_selfloops(G))

# number of first five strongly connected components
strongly_connected_components = nx.strongly_connected_components(G)
first_components = sorted(strongly_connected_components, key=len, reverse=True)[:5]

# in strengths and node percentage - change dict parameters for other measures
in_strengths = dict(G.out_degree(weight="weight"))
in_strength_counts = {}
for strength in in_strengths.values():
    if strength in in_strength_counts:
        in_strength_counts[strength] += 1
    else:
        in_strength_counts[strength] = 1

num_nodes = len(G.nodes)
in_strength_percs = {k: (v/num_nodes)*100 for k, v in in_strength_counts.items()}
scatter_data = [(k, v) for k, v in in_strength_percs.items()]

'''
plt.scatter([d[0] for d in scatter_data], [d[1] for d in scatter_data])
plt.yscale("log")
plt.xlim([0, 20])
plt.ylim([0.01, 100])
plt.title('Out-strength and nodes percentage')
plt.xlabel('Out-strength')
plt.ylabel('Nodes percentage')
#plt.show()
'''

# nodes percentage which have the in link coming from the most 5 out degree nodes
nodes_with_one_in_link = [n for n in G.nodes() if G.in_degree(n) == 1]
outdegree = dict(G.out_degree())
top_nodes = sorted(outdegree, key=outdegree.get, reverse=True)[:5]

selected_edges = [(u, v) for (u, v) in G.edges() if u in top_nodes]
print(len(selected_edges))
selected_graph = nx.DiGraph(selected_edges)
nodes_with_one_incoming_edge_from_out_degree_hubs = [n for n in selected_graph.nodes() if selected_graph.in_degree(n) == 1]

print("Percentage = " + str((len(nodes_with_one_incoming_edge_from_out_degree_hubs)*100)/(len(nodes_with_one_in_link))))

# Dimension of the largest strongly connected component
scc = max(nx.strongly_connected_components(G), key=len)
print("Dimension of the largest strongly connected component: " + str(len(scc)))

# Number of all the strongly connected components
n_scc = nx.number_strongly_connected_components(G)
print("Number of all the strongly connected components (SCC): " + str(n_scc))

# Dimension of the IN and OUT components
sccs = list(nx.strongly_connected_components(G))
scc_subgraphs = (G.subgraph(c) for c in nx.strongly_connected_components(G))
largest_scc = max(scc_subgraphs, key=len)
in_component = set()
out_component = set()

for node in largest_scc.nodes():
    predecessors = G.predecessors(node)
    in_component.update(pred for pred in predecessors if pred not in largest_scc)

for node in largest_scc.nodes():
    successors = G.successors(node)
    out_component.update(succ for succ in successors if succ not in largest_scc)


out_component_new = out_component.copy()
while(True):
    out_component_new = out_component.copy()

    for node in out_component_new:
        successors = G.successors(node)
        out_component.update(succ for succ in successors if succ not in largest_scc)

    if out_component == out_component_new:
        break

print("Dimension of the largest IN component (IN): " + str(len(in_component)))
print("Dimension of the largest OUT component (OUT): " + str(len(out_component)))

# Average shortest path length
avg_path = nx.average_shortest_path_length(largest_scc)
print("Average shortest path length in SCC: " + str(avg_path))

"""
max_shortest = 0
for node_a in G.nodes():
    for node_b in G.nodes():
        if node_a == node_b:
            continue
        try: 
            shortest = len(nx.shortest_path(G, source=node_a, target=node_b)) - 2
            if shortest > max_shortest:
                max_shortest = shortest
        except:
            continue    
"""
# calcola i shortest path
shortest_paths = nx.shortest_path(largest_scc)
path_lengths = {}
total_paths = 0
for source in shortest_paths:
    for target in shortest_paths[source]:
        print(shortest_paths[source][target])
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
plt.xlabel('Shortest Path Length')
plt.ylabel('Nodes percentage')
plt.show()