function main()
    clc; clear; close all;

    % Define matrix sizes
    ns = 100:100:1000;
    
    % Storage for timing results
    times_inv = zeros(length(ns), 1);
    times_lu = zeros(length(ns), 1);
    times_chol = zeros(length(ns), 1);

    fprintf('Running simulations...\n');
    fprintf('%-10s %-15s %-15s %-15s\n', 'n', 'Inverse(s)', 'LU(s)', 'Cholesky(s)');

    for k = 1:length(ns)
        n = ns(k);
        
        % 1. Generate data as per remarks
        alpha = randn(n, 1);
        b = randn(n, 1);
        I = eye(n);
        A = I + alpha * alpha';
        
        % 2. Method 1: Inverse Method
        tic;
        x_inv = solve_by_inverse(A, b);
        times_inv(k) = toc;
        
        % 3. Method 2: LU Decomposition
        tic;
        x_lu = solve_by_lu(A, b);
        times_lu(k) = toc;
        
        % 4. Method 3: Cholesky Decomposition
        tic;
        x_chol = solve_by_cholesky(A, b);
        times_chol(k) = toc;

        % Print progress
        fprintf('%-10d %-15.4f %-15.4f %-15.4f\n', n, times_inv(k), times_lu(k), times_chol(k));
    end

    % Plotting results
    figure;
    plot(ns, times_inv, '-o', 'LineWidth', 2, 'DisplayName', 'Inverse Method');
    hold on;
    plot(ns, times_lu, '-s', 'LineWidth', 2, 'DisplayName', 'LU Decomposition');
    plot(ns, times_chol, '-^', 'LineWidth', 2, 'DisplayName', 'Cholesky Decomposition');
    hold off;
    
    xlabel('Matrix Size (n)');
    ylabel('Time (seconds)');
    title('Performance Comparison: Solving Ax=b');
    legend('Location', 'NorthWest');
    grid on;
end


% Method 1: Inverse via Gauss-Jordan
function x = solve_by_inverse(A, b)
    n = size(A, 1);
    % Augment A with Identity matrix
    M = [A, eye(n)];
    
    % Forward Elimination
    for i = 1:n
        % Normalize pivot row
        pivot = M(i, i);
        M(i, :) = M(i, :) / pivot;
        
        % Eliminate other rows
        for k = 1:n
            if k ~= i
                factor = M(k, i);
                M(k, :) = M(k, :) - factor * M(i, :);
            end
        end
    end
    
    % Extract Inverse
    A_inv = M(:, n+1:end);
    
    % Compute x = A^-1 * b
    x = A_inv * b;
end

% Method 2: LU Decomposition (Doolittle Algorithm)
function x = solve_by_lu(A, b)
    n = size(A, 1);
    L = eye(n);
    U = zeros(n);
    
    % Decompose A = LU
    for j = 1:n
        % Upper Triangular
        for i = 1:j
            sum_k = 0;
            for k = 1:i-1
                sum_k = sum_k + L(i, k) * U(k, j);
            end
            U(i, j) = A(i, j) - sum_k;
        end
        
        % Lower Triangular
        for i = j+1:n
            sum_k = 0;
            for k = 1:j-1
                sum_k = sum_k + L(i, k) * U(k, j);
            end
            L(i, j) = (A(i, j) - sum_k) / U(j, j);
        end
    end
    
    % Forward Substitution (Ly = b)
    y = zeros(n, 1);
    for i = 1:n
        y(i) = b(i) - L(i, 1:i-1) * y(1:i-1);
    end
    
    % Backward Substitution (Ux = y)
    x = zeros(n, 1);
    for i = n:-1:1
        x(i) = (y(i) - U(i, i+1:n) * x(i+1:n)) / U(i, i);
    end
end

% Method 3: Cholesky Decomposition
function x = solve_by_cholesky(A, b)
    n = size(A, 1);
    G = zeros(n);
    
    % Decompose A = G * G^T
    for j = 1:n
        sum_k = 0;
        if j > 1
             sum_k = sum(G(j, 1:j-1).^2);
        end
        
        G(j, j) = sqrt(A(j, j) - sum_k);
        
        for i = j+1:n
            sum_k_ij = 0;
            if j > 1
                sum_k_ij = sum(G(i, 1:j-1) .* G(j, 1:j-1));
            end
            G(i, j) = (A(i, j) - sum_k_ij) / G(j, j);
        end
    end
    
    % Forward Substitution (Gy = b)
    y = zeros(n, 1);
    for i = 1:n
        y(i) = (b(i) - G(i, 1:i-1) * y(1:i-1)) / G(i, i);
    end
    
    % Backward Substitution (G'x = y) -> Transpose G effectively
    GT = G'; 
    x = zeros(n, 1);
    for i = n:-1:1
        x(i) = (y(i) - GT(i, i+1:n) * x(i+1:n)) / GT(i, i);
    end
end