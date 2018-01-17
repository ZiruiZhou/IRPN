function [funv_array, x_array, tt, resi_array, num_iter, cput, ssa] = alg_rpn_tt( x0, data, model, opts)

% This function uses IRPN to solve l1-regularized logistic regression
% problem. The decription of the algorithm can be found in our paper:

% M.-C. Yue, Z. Zhou, A. M.-C. So:

% "A Family of Inexact SQA Methods for Non-Smooth Convex Minimization with 
% Provable Convergence Guarantees Based on the Luo-Tseng Error Bound
% Property"

% Inputs:
%   1. x0: initial point
%   2. data: samples and labels
%   3. model: regpara, penalty, loss, eps
%   4. opts: IRPN specific parameters:
%           (a). opts.maxit: maximum number of outer iterations
%           (b). opts.maxitsub: maximum number of inner iterations
%           (c). opts.mu = 1e-6
%           (d). opts.eta = 0.5;
%           (e). opts.rho = 0, 0.5, or 1.


% Outputs:
%   1. funv_array: the array of function values of each outer iterate
%   2. resi_array: the array of residual values of each outer iterate
%   3. num_iter: the array of every number of inner iterations
%   4. cput: cpu time
%   5. ssa: the array of step sizes of each iteration



x = x0;

A = data.A;
b = data.b;
[m,n] = size(A);
eps = model.eps;
rp = model.regpara;

sigma = opts.sigma;
beta = opts.beta;
maxit = opts.maxit;
maxitsub = opts.maxitsub;
mu = opts.mu;
eta = opts.eta;
rho = opts.rho;

% record stepsizes, function values, residuals and number of each inner
% iterations
ssa = zeros(1,maxit);
funv_array = zeros(1, maxit);
resi_array = zeros(1, maxit);
num_iter = zeros(1, maxit);

x_array = zeros(n,100);
tt = zeros(1,100);
ti = 1;
count = 0;
x_array(:,ti) = x;

tstart = tic;

for k=1:maxit-1
    
    %
    % Computing function value, gradient, hessian diagonal, and residual at x
    %
    
    funv_s = loss_smv(x,data);
    funv_n = nsm_funv(x,model);
    funv = funv_s + funv_n;
    funv_array(1,k) = funv;
    
    grad = loss_smg(x,data);
    Dx = loss_smh_diag(x,data);
    
    resi = norm(x - prox_oper_vec(x-grad,model,1),2);
    resi_array(k) = resi;
    
%     fprintf('At iteration %d of RPN_CD, residual %f\n', k, resi);
    if resi<eps 
        fprintf('IRPN rho = %1.1f achieves required accuracy\n',rho);
        break;
    end
    


    
    %
    % perform regularized Newton with line search
    %
    
    mu = mu*max(resi, (resi)^rho);    % Hessian regularization
    eps_in = eta*min(resi, resi^(1+rho));    % stopping cri for inner loop
    
    s = x;
    As = A*x;
    q =  A'*(Dx.*(As)) + mu*x - grad;
    
    % The subproblem then can be written as
    %       min_x 0.5*s^T(A'*Dx*A+mu*I)s - q^Ts + lambda*||s||_1.
    resi_in = resi;
    
    for i=1:maxitsub
        if resi_in<eps_in
%             fprintf('inner finished with resi_in = %f\n', resi_in);
            break;
        end
        
        [s,As] = ripn_cordmin(s, A, Dx, As, q, rp, mu); % perform a cycle of coordinate minimization

        resi_in = norm(s - prox_oper_vec(s - (A'*(Dx.*(As)) + mu*s - q), model, 1), 2);
        
%         if mod(i-1,5)==0
%             eAx = exp(As);
%             grad_t = A'*(eAx./(1+eAx) - b)/m; 
%             resi_t = norm(s - prox_oper_vec(s-grad_t,model,1),2);
%             if resi_t<eps
%                 fprintf('inner finished with resi_in = %f\n', resi_in);
%                 break;
%             end
%         end
        count = count + 1;
        if mod(count,3)==0
            ti = ti + 1;
            x_array(:,ti) = s;
            tt(ti) = toc(tstart);
        end
    end
    
%========================================================================================
%   Below is pure Matlab implementation. It is much slower than Mex file
%   implementation. 
% 
%     p = size(A,2);
%     s = x;      % s is initial point of inner loop;
%     Ax = A*x;
%     As = Ax;
%     adax = A'*(Dx.*Ax);
%     hess_diag = zeros(p,1);
%     for i=1:p
%         hess_diag(i) = dot(A(:,i).^2, Dx);
%     end
%     
%     resi_in = resi;
%     fprintf('resi_in = %f\n', resi);
%     
%     for i=1:maxitsub
%         
%         if resi_in<eps_in
%             %fprintf('inner finished with resi_in = %f\n', resi_in);
%             break;
%         end
%         
%         for ind=1:p
%             aii = hess_diag(ind) + mu;
%             alpha = dot(A(:,ind), Dx.*As) - hess_diag(ind)*s(ind) + grad(ind) - adax(ind) - mu*x(ind);
%             s_p = soft_skg(-alpha, rp)/aii;
%             As = As + A(:,ind)*(s_p - s(ind));
%             s(ind) = s_p;
%         end
%         
%         adas = A'*(Dx.*As);
%         resi_in = norm(s - soft_skg(s - (grad + adas - adax + mu*(s - x)), rp), 2)
%     end
%========================================================================================
    
    
    % search direction d = s-x, and perform backtracking line search
    d = s - x;
    ss = 1;
    disc = ss*dot(grad, d) + nsm_funv(s,model) - funv_n;
    funv_new = loss_smv(s,data) + nsm_funv(s,model);
    while(funv_new > funv_s + funv_n + sigma*disc)
        ss = ss * beta;
        s = x + ss*d;
        disc = ss*dot(grad, d) + nsm_funv(s,model) - funv_n;
        funv_new = loss_smv(s,data) + nsm_funv(s,model);
    end
    
    ssa(k) = ss;
    num_iter(k) = i;
    
    x = s;

    ti = ti + 1;
    x_array(:,ti) = s;
    tt(ti) = toc(tstart);

%     fprintf('Iteration %d is finished with %d number of inner iterations\n\n',k,i-1);
end

xopt = x;
cput = toc(tstart);

end
    

