clc; clear; close all;

% 1. Define the matrices given in the problem
A = [0, 1; 
    -2, -3];

G = [1, 0; 
     0, 1];

F = [0, 0; 
     0, 1];

% 2. Call the custom function to solve the ARE
disp('Solving ARE using Schur Decomposition method...');
X = solve_are_general(A, G, F);

% 3. Display the results
disp('------------------------------------------------');
disp('Computed Solution X (Symmetric & Positive Definite):');
disp(X);

% 4. Verification (Check residual)
% The residual should be very close to zero (e.g., < 1e-14)
Residual = G + X*A + A'*X - X*F*X;
disp('Residual Norm (Should be close to 0):');
disp(norm(Residual));


% ==========================================
% Function Definition
% ==========================================
function X = solve_are_general(A, G, F)
    % Get the dimension n
    n = size(A, 1);
    
    % 1. Construct the Hamiltonian Matrix
    % NOTE: In standard ARE theory (Laub's method), the bottom-right block
    % is -A'. The prompt image showed A', which is likely a typo.
    % The Hamiltonian must have eigenvalues symmetric with respect to the imaginary axis.
    H = [A,   -F; 
        -G,  -A']; 
    
    % 2. Perform Schur Decomposition
    % H = U * T * U' where U is unitary and T is upper triangular (or quasi-upper triangular)
    [U, T] = schur(H);
    
    % 3. Reorder Eigenvalues
    % We need the stabilizing solution, which corresponds to the stable invariant subspace.
    % Therefore, we move eigenvalues with negative real parts (LHP) to the top-left block.
    % 'lhp' stands for Left Half Plane.
    [U_ord, T_ord] = ordschur(U, T, 'lhp');
    
    % 4. Extract Submatrices and Solve for X
    % According to invariant subspace theory, the stable subspace is spanned by the first n columns of U.
    % Let U = [U11; U21]. Then X * U11 = U21.
    U11 = U_ord(1:n, 1:n);
    U21 = U_ord(n+1:2*n, 1:n);
    
    % Compute X = U21 * inv(U11)
    % Using the slash operator (/) for better numerical stability than inv()
    X = U21 / U11;
    
    % 5. Enforce Symmetry (Remove numerical noise)
    % Ideally X is symmetric, but small numerical errors (approx 1e-15) might exist.
    X = real((X + X') / 2);
    
    % ==========================================
    % Plotting Section
    % ==========================================
    
    % Compute eigenvalues for verification
    eig_H = eig(H);          % Eigenvalues of the Hamiltonian Matrix
    eig_cl = eig(A - F*X);   % Eigenvalues of the Closed-Loop System (A - FX)
    
    figure('Name', 'Eigenvalue Distribution', 'Color', 'w');
    
    % Subplot 1: Eigenvalues of A - FX
    subplot(1, 2, 1);
    plot(real(eig_cl), imag(eig_cl), 'bo', 'MarkerSize', 8, 'LineWidth', 2);
    grid on; axis equal;
    xline(0, 'k--'); yline(0, 'k--');
    title('Eigenvalues of A - FX (Stable)');
    xlabel('Real Axis'); ylabel('Imaginary Axis');
    subtitle('Should be in Left Half Plane');
    
    % Subplot 2: Eigenvalues of Hamiltonian Matrix H
    subplot(1, 2, 2);
    plot(real(eig_H), imag(eig_H), 'rx', 'MarkerSize', 8, 'LineWidth', 2);
    grid on; axis equal;
    xline(0, 'k--'); yline(0, 'k--');
    title('Eigenvalues of Hamiltonian Matrix');
    xlabel('Real Axis'); ylabel('Imaginary Axis');
    subtitle('Should be symmetric about imaginary axis');
end