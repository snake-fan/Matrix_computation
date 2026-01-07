function main()
    clc; clear; close all;
    
    %% --- Part 1: Efficiency Advantage (Question 1) ---
    
    sizes = [1000, 3000, 5000]; 
    k_target = 50;  % Target rank
    p = 10;         % Oversampling
    q = 2;          % Power iterations
    
    t_classical = zeros(length(sizes), 1);
    t_rsvd = zeros(length(sizes), 1);
    
    for i = 1:length(sizes)
        n = sizes(i);
        m = n;
        
        fprintf('Processing Matrix Size: %dx%d... ', m, n);
        
        % Generate random dense matrix
        X = rand(m, n);
        
        % 1. Measure Classical SVD Time
        % Using 'econ' to ensure fair comparison for low-rank goals
        tic;
        [~, ~, ~] = svd(X, 'econ'); 
        t_classical(i) = toc;
        
        % 2. Measure Randomized SVD Time [cite: 16]
        tic;
        [~, ~, ~] = rSVD(X, k_target, p, q);
        t_rsvd(i) = toc;
        
        fprintf('Done. (Classical: %.2fs, rSVD: %.2fs)\n', t_classical(i), t_rsvd(i));
    end
    
    % Compute Speedup Ratio [cite: 17]
    speedup = t_classical ./ t_rsvd;
    
    % Display Efficiency Table
    fprintf('\n--- Efficiency Results ---\n');
    T = table(sizes', t_classical, t_rsvd, speedup, ...
        'VariableNames', {'MatrixSize', 'Classical_Sec', 'rSVD_Sec', 'Speedup_Ratio'});
    disp(T);
    
    % Plot Matrix Size vs Runtime [cite: 17]
    figure('Name', 'Q1: Efficiency Advantage');
    plot(sizes, t_classical, '-o', 'LineWidth', 2, 'DisplayName', 'Classical SVD');
    hold on;
    plot(sizes, t_rsvd, '-s', 'LineWidth', 2, 'DisplayName', 'Randomized SVD');
    xlabel('Matrix Dimension (m=n)');
    ylabel('Wall-clock Time (seconds)');
    legend('Location', 'northwest');
    title(sprintf('Efficiency: rSVD vs Classical SVD (k=%d, p=%d, q=%d)', k_target, p, q));
    grid on;
    drawnow;
    
    
    %% --- Part 2: Accuracy and Parameter Sensitivity (Question 2) ---
    
    % Load Image (We use standard 512x512 cameraman.tif) [cite: 19]
    try
        img = imread('cameraman.tif');
    catch
        warning('cameraman.tif not found. Please ensure the image is in the path.');
        return;
    end
    
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    A = im2double(img); % Convert to double [0, 1]
    [m, n] = size(A);
    
    % Parameters for Q2 [cite: 20, 21, 22]
    k_target = 30;              % Fixed target rank
    p_values = [0, 5, 10, 20];  % Oversampling
    q_values = [0, 1, 2, 3];    % Power iterations
    
    % Pre-calculate norms for error metrics
    norm_A = norm(A, 'fro'); 
    L = 1; % Max pixel value for double is 1
    
    % Storage for best/worst visualization
    max_psnr = -inf;
    min_psnr = inf;
    best_img = [];
    worst_img = [];
    best_setting = '';
    worst_setting = '';
    
    fprintf('%-5s %-5s %-15s %-10s\n', 'p', 'q', 'Rel.Frob.Error', 'PSNR(dB)');
    fprintf('----------------------------------------\n');
    
    for p = p_values
        for q = q_values
            % Run rSVD
            [U, S, V] = rSVD(A, k_target, p, q);
            
            % Reconstruct Image A_k = U*S*V'
            A_k = U * S * V';
            
            % 1. Relative Frobenius Error [cite: 24]
            % ||A - A_k||_F / ||A||_F
            diff_norm = norm(A - A_k, 'fro');
            rel_error = diff_norm / norm_A;
            
            % 2. PSNR Calculation [cite: 25, 26, 29]
            % MSE = (1/mn) * ||A - A_k||_F^2
            mse = (diff_norm^2) / (m * n);
            psnr_val = 10 * log10((L^2) / mse);
            
            fprintf('%-5d %-5d %-15.4e %-10.2f\n', p, q, rel_error, psnr_val);
            
            % Track Best and Worst [cite: 31]
            if psnr_val > max_psnr
                max_psnr = psnr_val;
                best_img = A_k;
                best_setting = sprintf('p=%d, q=%d', p, q);
            end
            
            if psnr_val < min_psnr
                min_psnr = psnr_val;
                worst_img = A_k;
                worst_setting = sprintf('p=%d, q=%d', p, q);
            end
        end
    end
    
    % Visualize Best and Worst Reconstructions [cite: 31]
    figure('Name', 'Q2: Reconstruction Quality');
    
    subplot(1, 3, 1);
    imshow(A); 
    title('Original Image');
    
    subplot(1, 3, 2);
    imshow(worst_img); 
    title({['Worst: ' worst_setting], ['PSNR: ' num2str(min_psnr, '%.2f') ' dB']});
    
    subplot(1, 3, 3);
    imshow(best_img); 
    title({['Best: ' best_setting], ['PSNR: ' num2str(max_psnr, '%.2f') ' dB']});

end

%% --- Randomized SVD Function Implementation ---
function [U, S, V] = rSVD(X, k, p, q)
    % Inputs:
    % X: Input matrix (m x n)
    % k: Target rank
    % p: Oversampling parameter
    % q: Number of power iterations
    
    [m, n] = size(X);
    l = k + p; % Sketch size [cite: 10]
    
    % Step 1: Generate Gaussian test matrix Omega (n x l) [cite: 11]
    Omega = randn(n, l);
    
    % Step 2: Compute Y = X * Omega [cite: 11]
    Y = X * Omega;
    
    % Step 3: Power Iterations with QR stabilization [cite: 12]
    % Perform q times to refine the range approximation
    for i = 1:q
        [Q_temp, ~] = qr(Y, 0);
        [Q_temp, ~] = qr(X' * Q_temp, 0); % Multiply by X' to maintain dimensions
        Y = X * Q_temp;
    end
    
    % Step 4: Orthogonalize Y to form Q [cite: 12]
    [Q, ~] = qr(Y, 0);
    
    % Step 5: Form smaller matrix B = Q' * X [cite: 13]
    B = Q' * X;
    
    % Step 6: Compute SVD of small matrix B [cite: 13]
    [U_hat, S, V] = svd(B, 'econ');
    
    % Step 7: Compute final singular vectors U = Q * U_hat [cite: 13]
    U = Q * U_hat;
    
    % Truncate to the exact target rank k
    % (Implicitly required for accurate low-rank error measurement in Q2)
    U = U(:, 1:k);
    S = S(1:k, 1:k);
    V = V(:, 1:k);
end