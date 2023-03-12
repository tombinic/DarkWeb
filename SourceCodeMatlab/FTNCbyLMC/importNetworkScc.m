data = readtable('C:\Users\nicol\Documents\GitHub\DarkWeb\Dataset\darkweb_scc.csv');

% Estrai i nodi e gli archi dal dataframe

nodes = unique([data.Source; data.Target]);
edges = table2array(data(:, [1, 2]));

weights = data.Weight;

G = digraph(edges(:, 1), edges(:, 2), weights);

num_nodes = numnodes(G);
num_edges = numedges(G);

%disp(G.Nodes);
A = adjacency(G, 'weighted');
for i = 1:num_nodes
    fprintf('Node name: %s, Node index: %d\n', G.Nodes.Name{i}, i);
end
save('A_darkweb_scc.mat', 'A');
