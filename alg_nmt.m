function [funv_array, xopt, resi_array, num_iter, cput, ssa] = alg_nmt( x0, data, model, opts )

% This function uses SpaRSA to solve l1-regularized logistic regression
% problem. The decription of the algorithm can be found in our paper:

% M.-C. Yue, Z. Zhou, A. M.-C. So:

% "A Family of Inexact SQA Methods for Non-Smooth Convex Minimization with 
% Provable Convergence Guarantees Based on the Luo-Tseng Error Bound
% Property"


% Inputs:
%   1. x0: initial point
%   2. data: samples and labels
%   3. model: regpara, penalty, loss, eps
%   4. opts: SpaRSA specific parameters:
%           (a). opts.maxit: maximum number of iterations
%           (b). opts.c = 1e-4;
%           (c). opts.M = 5;
%           (d). opts.ssu = 1e+8;
%           (e). opts.ssl = 1e-8;


% Outputs:
%   1. funv_array: the array of function values of each iterate
%   2. resi_array: the array of residual values of each iterate
%   3. num_iter: the number of iterations
%   4. cput: cpu time
%   5. ssa: the array of step-sizes used in each iteration


eps = model.eps;
beta = opts.beta;
maxit = opts.maxit;
ssu = opts.ssu;
ssl = opts.ssl;
c = opts.c;
M = opts.M;

dis_freq = Inf;
fprintf('SpaRSA is working...\n');

% record stepsizes, function values, and residuals
ssa = zeros(1, maxit);
funv_array = zeros(1,maxit);
resi_array = zeros(1,maxit);

tstart = tic;

k = 0;
x = x0;
ss0 = max(ssl,min(ssu,1));
funv_array(1) = loss_smv(x,data) + nsm_funv(x,model);
grad = loss_smg(x,data);
resi = norm(x - prox_oper_vec(x-grad,model,1),2);
resi_array(1) = resi;

F_cache = -inf(M,1);
F_cache(1) = funv_array(1);
Fmax = max(F_cache);

while k<maxit

    if mod(k,dis_freq)==0
        fprintf('This is teration %d of SpaRSA, and residual = %f\n', k, resi);
     end
        
    if resi<eps
        fprintf('SpaRSA achieves required accuracy\n');
        break;
    end
    
    %
    % perform a nonmonotone line search step
    %
    
    ss = ss0;
    while((ss>=ssl)&&(ss<=ssu))
        x_new = prox_oper_vec(x-ss*grad, model, ss);
        F_new = loss_smv(x_new,data) + nsm_funv(x_new,model);
        if F_new <= Fmax - c/2*norm(x_new - x)^2
            break
        end
        ss = ss*beta;
    end
    
    ssa(k+1) = ss;    % record the step size of this iteration
    
    % BB step size
    dx = x_new - x;
    grad_new = loss_smg(x_new,data);
    dg = grad_new - grad;
    chs1 = norm(dx)^2/abs(dx'*dg);
    chs2 = abs(dx'*dg)/norm(dg)^2;
    if mod(k,2) == 0
        ss0 = max(ssl,min(ssu,chs1));
    else
        ss0 = max(ssl,min(ssu,chs2));
    end
    
    % update iterate
    k = k+1;
    x = x_new;
    funv_array(k+1) = F_new;
    grad = grad_new;
    resi = norm(x - prox_oper_vec(x-grad,model,1),2);
    resi_array(k+1) = resi;
    F_cache(mod(k+1,M)+1,1) = F_new;
    Fmax = max(F_cache);
    
end

num_iter = k;
xopt = x;
cput = toc(tstart);

fprintf('SpaRSA is finished\n\n');

end



