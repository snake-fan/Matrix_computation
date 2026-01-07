%% Problem 1: Power Method with Deflation
clear; clc; close all;

% 1. Construct the matrix
target_eigs = [1, 2, 3, 4, 5];
n = length(target_eigs);
Lambda = diag(target_eigs);

% Generate random orthogonal matrix X
[Q, ~] = qr(randn(n)); 
A = Q * Lambda * Q'; % A is symmetric

% 2. Power Method with Deflation
found_eigs = zeros(n, 1);
errors = [];
iteration_counts = [];

% We need to copy A because we will deflate it
A_curr = A;
max_iter = 20;
tol = 1e-10;

figure; hold on;
colors = {'r', 'g', 'b', 'm', 'k'};

for k = 1:n
    % Initialize random vector
    v = randn(n, 1);
    v = v / norm(v);
    
    err_history = [];
    
    % The true eigenvalue we are looking for (descending order)
    true_eig = target_eigs(n - k + 1); 
    
    for iter = 1:max_iter
        % Power step
        w = A_curr * v;
        lambda = v' * w; % Rayleigh Quotient
        v_next = w / norm(w);
        
        % Check convergence against the TRUE eigenvalue for plotting
        err = abs(lambda - true_eig);
        err_history = [err_history, err];
        
        % Stopping rule
        if norm(v_next - v) < tol
            break;
        end
        v = v_next;
    end
    
    % Store result
    found_eigs(k) = lambda;
    
    % Plot convergence for this eigenvalue
    semilogy(1:length(err_history), err_history, ...
        'DisplayName', sprintf('Eig %d', true_eig), ...
        'Color', colors{k}, 'LineWidth', 1.5);
    
    % Deflate the matrix: A_new = A - lambda * v * v'
    A_curr = A_curr - lambda * (v * v');
end

title('Convergence of Power Method with Deflation');
xlabel('Iteration');
ylabel('Error |\lambda_{calc} - \lambda_{true}|');
legend('show');
grid on;

fprintf('True Eigenvalues: %s\n', mat2str(sort(target_eigs, 'descend')));
fprintf('Found Eigenvalues: %s\n', mat2str(found_eigs'));