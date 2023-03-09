data = readtable('C:\Users\nicol\Documents\GitHub\DarkWeb\Dataset\darkweb_scc.csv');

% Estrai i nodi e gli archi dal dataframe

nodes = unique([data.Source; data.Target]);
edges = table2array(data(:, [1, 2]));

weights = data.Weight;

G = digraph(edges(:, 1), edges(:, 2), weights);

num_nodes = numnodes(G);
num_edges = numedges(G);

A = adjacency(G);

save('A_darkweb_scc_unweighted.mat', 'A');
