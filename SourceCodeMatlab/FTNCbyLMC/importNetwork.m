data = readtable('C:\Users\nicol\Documents\GitHub\DarkWeb\Dataset\darkweb.csv');

% Estrai i nodi e gli archi dal dataframe
nodes = unique([data.Source; data.Target]);
edges = table2array(data(:, [1, 2]));

weights = data.Weight;
G = graph(edges(:, 1), edges(:, 2), weights);

num_nodes = numnodes(G);
num_edges = numedges(G);

A = adjacency(G);

%[coreness, kn] = kcoreness_centrality_bu(A)
%histogram(coreness)
%p = plot(G, 'NodeColor', 'g', 'MarkerSize', max(normalize(coreness)));
%p.NodeCData = coreness;

save('A_darkweb.mat', 'A');