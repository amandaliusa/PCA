clear; clc; close all;
data = load('pca_data.mat');

%{
a. Find the m × m covariance matrix of X.
%}

X = data.X;
centroid = mean(X, 2);

for i=1:10000
    X(:,i) = X(:,i) - centroid; % preprocess X to make zero-sum rows 
end

cX = X * X' / 10000; % find covariance matrix

%{
b. Find m principal components of X and show them on top of the original
data. That is, if v is a principle component, then plot line l(t) = µ + vt
over the scatter plot in Figure 1, where µ in R2
is the centroid of the data.
%}

[V, D] = eig(cX);           % V is matrix of principal components 
t = linspace(-22, 20);
l1 = centroid + V(:,1) * t; % line corresponding to first PC
l2 = centroid + V(:,2) * t; % line corresponding to second PC


figure;
plot(data.X(1,:), data.X(2,:), '.'); % plot original data
hold on;
line(l1(1,:), l1(2,:), 'Color', 'red'); % line corresponding to first PC
hold on;
line(l2(1,:), l2(2,:), 'Color', 'red'); % line corresponding to second PC
xlim([4, 17]);
ylim([3, 15]);
xlabel('x1');
ylabel('x2');
legend('original data', 'principal components');

%{
c. If we change the basis of Rm from the standard basis to the basis 
consisting of the principal components [v1, . . . , vm] = V of X, the data 
matrix will transform to Y = V^-1X. Find the covariance of the transformed 
data Y .
%}

Y = inv(V) * X;      % transformed data 
cY = Y * Y' / 10000; % covariance matrix of Y

%{
d. Principal component analysis is a very popular data analysis technique 
and it is often used in various applications. As such, it is implemented in 
most numerical software packages. In Matlab, this is a built-in function 
pca. In the simplest form, V = pca(X) returns the principal components the
data matrix X, where rows of X correspond to measurements and columns
correspond to features. Find the principal components of X using pca and
make sure that they are the same as those you found in (b).
%}

V2 = pca(X'); 
% Indeed, V2 differs from V in part (b) only by a constant and the 
% order of the eigenvectors; it would give the same lines as in part (b) 
% if graphed.