%% Problem 2: QR Iteration (Shifted vs Unshifted)
clear; clc;

% Construct Matrix
target_eigs = [1, 5, 9, 11, 14];
n = length(target_eigs);
[Q_rot, ~] = qr(randn(n));
A = Q_rot * diag(target_eigs) * Q_rot';

% Target eigenvalue (usually largest or smallest depending on convergence)
% With basic QR, A(n,n) usually converges to the smallest eigenvalue first 
% if we don't sort, but generally it converges to *one* of them.
% We calculate error relative to the closest true eigenvalue.

max_iter = 50;

% --- Method 1: Unshifted ---
A_unshift = A;
err_unshift = [];
for k = 1:max_iter
    [Q, R] = qr(A_unshift);
    A_unshift = R * Q;
    
    % Estimate is the bottom-right element
    est = A_unshift(n,n);
    % Find closest true eigenvalue to measure error
    [min_err, ~] = min(abs(target_eigs - est));
    err_unshift = [err_unshift, min_err];
end

% --- Method 2: Rayleigh Quotient Shift ---
A_shift = A;
err_shift = [];
for k = 1:max_iter
    sigma = A_shift(n,n); % Rayleigh shift
    
    [Q, R] = qr(A_shift - sigma * eye(n));
    A_shift = R * Q + sigma * eye(n);
    
    est = A_shift(n,n);
    [min_err, ~] = min(abs(target_eigs - est));
    err_shift = [err_shift, min_err];
    
    if min_err < 1e-12
        break;
    end
end

% Plotting
figure;
semilogy(1:length(err_unshift), err_unshift, 'b-', 'LineWidth', 2);
hold on;
semilogy(1:length(err_shift), err_shift, 'r--', 'LineWidth', 2);
title('QR Iteration Convergence Comparison');
xlabel('Iteration');
ylabel('Error to nearest eigenvalue');
legend('Unshifted QR', 'Rayleigh Shift QR');
grid on;