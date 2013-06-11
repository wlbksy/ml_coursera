function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

m = size(X,1);
idx = zeros(size(X,1), 1);

for i = 1:m
    x = X(i,:);
    norm_two = zeros(K,1);
    for j = 1:K
		delta = x - centroids(j,:);
		norm_two(j) = delta * delta';
    end
    [~, idx(i)] = min(norm_two);
end

end

