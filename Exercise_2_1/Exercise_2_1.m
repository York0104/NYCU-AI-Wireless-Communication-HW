% Exercise 2.1
% Estimate path-loss exponent gamma and shadowing variance sigma_phi_dB^2
% from empirical measurements of Pr/Pt at 900 MHz.

clear; clc;

%% Given data
f = 900e6;                    % frequency (Hz)
c = 3e8;                      % speed of light (m/s)
lambda = c / f;               % wavelength
d0 = 1;                       % reference distance (m)

d = [5; 30; 60; 110; 500];    % distances (m)
Pr_Pt_dB = [-60; -80; -105; -115; -135];   % measured Pr/Pt in dB

%% Step 1: Compute K from free-space model at d0
% 10*log10(K) = 20*log10(lambda/(4*pi*d0))
K_dB = 20*log10(lambda / (4*pi*d0));   % = 10*log10(K)
log10K = K_dB / 10;                    % = log10(K)

%% Step 2: Form xi and yi
x = log10(d / d0);
y = Pr_Pt_dB / 10;                     % y = log10(Pr/Pt)

%% Step 3: Least-squares estimation of gamma
% Correct model: y = log10K - gamma*x + epsilon
gamma_hat = -(x' * (y - log10K)) / (x' * x);

%% Step 4: Compute residuals
eps_hat = y - (log10K - gamma_hat * x);

%% Step 5: Shadowing standard deviation and variance
% Convert residuals to dB: phi_dB = 10 * epsilon
phi_hat_dB = 10 * eps_hat;

% MSE-based variance estimate
sigma_phi_dB_sq = mean(phi_hat_dB.^2);
sigma_phi_dB = sqrt(sigma_phi_dB_sq);

%% Display results
fprintf('Estimated path-loss exponent gamma = %.4f\n', gamma_hat);
fprintf('Estimated shadowing std dev sigma_phi_dB = %.4f dB\n', sigma_phi_dB);
fprintf('Estimated shadowing variance sigma_phi_dB^2 = %.4f dB^2\n', sigma_phi_dB_sq);

%% measured vs fitted values
y_fit = log10K - gamma_hat * x;
Pr_Pt_fit_dB = 10 * y_fit;

figure;
plot(log10(d), Pr_Pt_dB, 'o', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
plot(log10(d), Pr_Pt_fit_dB, '-s', 'LineWidth', 1.5, 'MarkerSize', 7);
grid on;
xlabel('log_{10}(distance in m)');
ylabel('P_r / P_t (dB)');
legend('Measured', 'Fitted', 'Location', 'best');
title('Path-loss fitting');