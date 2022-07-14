clc; clear all; close all;

N = 20; M = 40; D_0 = 7;  %introducing values
phi = randn(N, M);


w_in = randn(D_0, 1);
w = zeros(M, 1);
r = randi([1,M],D_0,1);
w(r) = w_in;    %True value of w

NMSE = zeros(5,1);

j = 1;
for sigma_sq_dB = -20:5:0
    
    sigma_sqr = 10 ^ (sigma_sq_dB/10);

    noise = sigma_sqr.* randn(20,1);
    t = phi * w + noise;    % our measurements t

    max_ite = 1000000; threshold = 10^(-3); ite = 1;error_ite = 0;

    alpha = 100;
    A = alpha * eye(40);
    posterior_mean_old = 0.001 * ones(40,1);

    while(1)
        posterior_var = inv(((1/sigma_sqr) * (phi' * phi)) + A);        %posterior variance is taken here
        posterior_mean = (1/sigma_sqr) * posterior_var * phi' * t;      %posterior mean is thus calculated

        for i = 1:M
            gamma(i) = 1 - A(i,i) * posterior_var(i,i);
            alpha_new(i) = gamma(i)/(posterior_mean(i) * posterior_mean(i));
        end
    
        A = diag(alpha_new);    %optimizing parameter values
    
        error_ite = (norm((posterior_mean - posterior_mean_old), 2))/(norm(posterior_mean_old, 2)); %updating anc calculate error
    
        if(error_ite <= threshold)
            break;
        end
    
        posterior_mean_old = posterior_mean;
    
    end
    w_est = posterior_mean;     % w as MAP estimate of posterior
    NMSE(j) = (norm((w_est - w), 2))/(norm(w, 2));      %calculating NMSE for each sigma
    j = j + 1;
end

semilogy([-20:5:0], NMSE)
xlabel("sigma square dB"); ylabel("NMSE");
title("MAP estimate of weights and NMSE vs variance plot")

