function main()
    % Initialization
    clc; clear; close all;
    
    dims = 2:20;
    n_dims = length(dims);
    
    % Pre-allocate arrays for results
    err_decomp_cgs = zeros(n_dims, 1);
    err_orth_cgs = zeros(n_dims, 1);
    
    err_decomp_mgs = zeros(n_dims, 1);
    err_orth_mgs = zeros(n_dims, 1);
    
    err_decomp_hh = zeros(n_dims, 1);
    err_orth_hh = zeros(n_dims, 1);
    
    err_decomp_givens = zeros(n_dims, 1);
    err_orth_givens = zeros(n_dims, 1);
    
    % Main Loop: Iterate over different dimensions n
    for k = 1:n_dims
        n = dims(k);
        A = generate_matrix(n);
        I = eye(n);
        
        % 1. Classical Gram-Schmidt (CGS)
        [Q, R] = qr_cgs(A);
        err_decomp_cgs(k) = norm(Q*R - A);      % Default norm() is the 2-norm (spectral norm)
        err_orth_cgs(k) = norm(Q'*Q - I);
        
        % 2. Modified Gram-Schmidt (MGS)
        [Q, R] = qr_mgs(A);
        err_decomp_mgs(k) = norm(Q*R - A);
        err_orth_mgs(k) = norm(Q'*Q - I);
        
        % 3. Householder Reflections
        [Q, R] = qr_householder(A);
        err_decomp_hh(k) = norm(Q*R - A);
        err_orth_hh(k) = norm(Q'*Q - I);
        
        % 4. Givens Rotations
        [Q, R] = qr_givens(A);
        err_decomp_givens(k) = norm(Q*R - A);
        err_orth_givens(k) = norm(Q'*Q - I);
    end
    
    % --- Plotting ---
    figure('Position', [100, 100, 1000, 500]);
    
    % Plot 1: Decomposition Accuracy ||QR - A||
    subplot(1, 2, 1);
    semilogy(dims, err_decomp_cgs, '-o', 'DisplayName', 'CGS'); hold on;
    semilogy(dims, err_decomp_mgs, '-s', 'DisplayName', 'MGS');
    semilogy(dims, err_decomp_hh, '-^', 'DisplayName', 'Householder');
    semilogy(dims, err_decomp_givens, '-d', 'DisplayName', 'Givens');
    xlabel('Dimension n');
    ylabel('Error ||QR - A||_2');
    title('Decomposition Accuracy');
    legend('Location', 'best');
    grid on;
    
    % Plot 2: Orthogonality Error ||Q^T Q - I||
    subplot(1, 2, 2);
    semilogy(dims, err_orth_cgs, '-o', 'DisplayName', 'CGS'); hold on;
    semilogy(dims, err_orth_mgs, '-s', 'DisplayName', 'MGS');
    semilogy(dims, err_orth_hh, '-^', 'DisplayName', 'Householder');
    semilogy(dims, err_orth_givens, '-d', 'DisplayName', 'Givens');
    xlabel('Dimension n');
    ylabel('Error ||Q^T Q - I_n||_2');
    title('Orthogonality Error');
    legend('Location', 'best');
    grid on;
end

% --- Helper Function: Generate Ill-Conditioned Matrix A ---
function A = generate_matrix(n)
    epsilon = 10^-6;
    e = ones(n, 1) / sqrt(n);
    alpha = e * sqrt(1 - epsilon);
    % A = In - alpha * alpha^T
    A = eye(n) - alpha * alpha';
end

% --- 1. Classical Gram-Schmidt (CGS) ---
function [Q, R] = qr_cgs(A)
    [m, n] = size(A);
    Q = zeros(m, n);
    R = zeros(n, n);
    for j = 1:n
        v = A(:, j);
        for i = 1:j-1
            % CGS projects onto the original A(:,j)
            R(i, j) = Q(:, i)' * A(:, j); 
            v = v - R(i, j) * Q(:, i);
        end
        R(j, j) = norm(v);
        Q(:, j) = v / R(j, j);
    end
end

% --- 2. Modified Gram-Schmidt (MGS) ---
function [Q, R] = qr_mgs(A)
    [m, n] = size(A);
    Q = zeros(m, n);
    R = zeros(n, n);
    v = A; % Work on a copy of A
    for i = 1:n
        R(i, i) = norm(v(:, i));
        Q(:, i) = v(:, i) / R(i, i);
        for j = i+1:n
            R(i, j) = Q(:, i)' * v(:, j);
            % MGS updates the remaining vectors immediately (in-place)
            % This reduces the propagation of round-off errors
            v(:, j) = v(:, j) - R(i, j) * Q(:, i); 
        end
    end
end

% --- 3. Householder Reflections ---
function [Q, R] = qr_householder(A)
    [m, n] = size(A);
    R = A;
    Q = eye(m);
    for k = 1:min(m-1, n)
        x = R(k:m, k);
        e1 = zeros(length(x), 1);
        e1(1) = 1;
        
        % Choose sign to avoid catastrophic cancellation: 
        % alpha = -sign(x(1)) * ||x||
        s = sign(x(1)); 
        if s == 0, s = 1; end % Handle case where x(1) is 0
        alpha_val = -s * norm(x);
        
        u = x - alpha_val * e1;
        v = u / norm(u); % Normalize the reflection vector
        
        % Update R: H * R
        % Using H = I - 2vv', update only the relevant submatrix
        R(k:m, k:n) = R(k:m, k:n) - 2 * v * (v' * R(k:m, k:n));
        
        % Update Q: Q * H
        % Q accumulates transformations; applied to columns
        Q(:, k:m) = Q(:, k:m) - 2 * (Q(:, k:m) * v) * v';
    end
end

% --- 4. Givens Rotations ---
function [Q, R] = qr_givens(A)
    [m, n] = size(A);
    R = A;
    Q = eye(m);
    
    for j = 1:n
        for i = m:-1:j+1
            a = R(i-1, j);
            b = R(i, j);
            
            % Compute Givens rotation parameters c (cos) and s (sin)
            if b == 0
                c = 1; s = 0;
            else
                % Numerically stable computation to avoid overflow/underflow
                if abs(b) > abs(a)
                    tau = -a / b;
                    s = 1 / sqrt(1 + tau^2);
                    c = s * tau;
                else
                    tau = -b / a;
                    c = 1 / sqrt(1 + tau^2);
                    s = c * tau;
                end
            end
            
            % Apply G^T to R (Row operation on rows i-1 and i)
            % G_T = [c -s; s c]
            G_T = [c, -s; s, c];
            R([i-1, i], j:n) = G_T * R([i-1, i], j:n);
            
            % Apply G to Q (Column operation on columns i-1 and i)
            % Update accumulation matrix Q
            Q(:, [i-1, i]) = Q(:, [i-1, i]) * G_T';
        end
    end
end