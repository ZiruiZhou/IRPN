function [funv_array, xopt, resi_array, num_iter, cput, Lipt] = alg_fista( x0, data, model, opts )

% This function uses FISTA to solve l1-regularized logistic regression
% problem. The decription of the algorithm can be found in our paper:

% M.-C. Yue, Z. Zhou, A. M.-C. So:

% "A Family of Inexact SQA Methods for Non-Smooth Convex Minimization with 
% Provable Convergence Guarantees Based on the Luo-Tseng Error Bound
% Property"

% Inputs:
%   1. x0: initial point
%   2. data: samples and labels
%   3. model: regpara, penalty, loss, eps
%   4. opts: fista specific parameters:
%           (a). opts.maxit: maximum number of iterations
%           (b). opts.adp: the indicator for using adaptive restart scheme (default = 0)


% Outputs:
%   1. funv_array: the array of function values of each iterate
%   2. resi_array: the array of residual values of each iterate
%   3. num_iter: the number of iterations
%   4. cput: cpu time
%   5. Lipt: cpu time for computing the Lipschitz constant


if ~isfield(opts, 'adp')
    adp = 0;
else
    adp = opts.adp;
end

dis_freq = Inf;     % Frequency for display information
fprintf('FISTA is working...\n');

eps = model.eps;
maxit = opts.maxit;


% record function values, and residuals
funv_array = zeros(1,maxit);
resi_array = zeros(1,maxit);

tstart = tic;

% computing the Lipschitz constant and 
A = data.A;
m = size(A,1);
ss = 4*m/eigs(A*A',1,'LM');

Lipt = toc(tstart);

k = 0;
x = x0;
xo = x0;
t = 1;
to = 1;
funv_array(1) = loss_smv(x,data) + nsm_funv(x,model);
grad = loss_smg(x,data);
resi = norm(x - prox_oper_vec(x-grad,model,1),2);
resi_array(1) = resi;

while k<maxit
    
    if mod(k,dis_freq)==0
        fprintf('This is iteration %d of FISTA, and residual = %f\n', k, resi);
     end
        
    if resi<eps
        fprintf('FISTA achieves required accuracy\n');
        break;
    end
    
    %
    % perform FISTA iteration
    %
    
    y = x + (to-1)/to*(x - xo);
    grad_y = loss_smg(y,data);
    x_new = prox_oper_vec(y-ss*grad_y, model, ss);
    F_new = loss_smv(x_new,data) + nsm_funv(x_new,model);
    
    
    if adp==1 && (y-x_new)'*(x_new-x)>0
%         fprintf('restarted\n');
        t = 1;
        to = 1;
    else
        to = t;
        t = (1+sqrt(1+4*t^2))/2;
    end
    
    
    % update iterate
    k = k+1;
    xo = x;
    x = x_new;
    funv_array(k+1) = F_new;
    grad = loss_smg(x,data);
    resi = norm(x - prox_oper_vec(x-grad,model,1),2);
    resi_array(k+1) = resi;
    
end

num_iter = k;
xopt = x;
cput = toc(tstart);

fprintf('FISTA is finished\n\n');

end


