function plotData(X, y)

%	PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;


hold on;
p = find(y==1);
n = find(y==0);

plot(X(p, 1), X(p, 2), 'k+');
plot(X(n, 1), X(n, 2), 'ko','MarkerFaceColor','y');


hold off;

end
