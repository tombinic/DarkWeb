function [shell] = k_shell(A,kmax)
% K_SHELL computes the k-shell decomposition of a graph represented by its adjacency matrix A.
%   kmax is the maximum shell to compute (default: maximum possible shell).
%   shell is a vector containing the k-shell index of each node.
%
%   Note: this function requires the Bioinformatics Toolbox.
%
%   Example:
%       A = [0 1 1 1 0 0;
%            1 0 1 0 0 0;
%            1 1 0 1 0 0;
%            1 0 1 0 1 1;
%            0 0 0 1 0 1;
%            0 0 0 1 1 0];
%       shell = k_shell(A);
%       plot(graph(A),'NodeCData',shell);

if nargin < 2
    kmax = max(sum(A));
end

B = A;
shell = zeros(size(A,1),1);

for k=1:kmax
    idx = find(sum(B)==k-1);
    if isempty(idx)
        break;
    end
    shell(idx) = k;
    B(idx,:) = 0;
    B(:,idx) = 0;
end
end