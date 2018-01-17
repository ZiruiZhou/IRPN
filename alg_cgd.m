function [funv_array, xopt, resi_array, num_iter, cput] = alg_cgd( x0, data, model, opts )

% This function uses CGD to solve l1-regularized logistic regression
% problem. The decription of the algorithm can be found in our paper:

% M.-C. Yue, Z. Zhou, A. M.-C. So:

% "A Family of Inexact SQA Methods for Non-Smooth Convex Minimization with 
% Provable Convergence Guarantees Based on the Luo-Tseng Error Bound
% Property"


% Inputs:
%   1. x0: initial point
%   2. data: samples and labels
%   3. model: regpara, penalty, loss, eps
%   4. opts: CGD specific parameters:
%           (a). opts.maxit: maximum number of iterations


% Outputs:
%   1. funv_array: the array of function values of each iterate
%   2. resi_array: the array of residual values of each iterate
%   3. num_iter: the number of iterations
%   4. cput: cpu time


eps = model.eps;
reg = model.regpara;
maxit = opts.maxit;

dis_freq = Inf;
fprintf('CGD is working...\n');

funv_array = zeros(1,maxit);
resi_array = zeros(1,maxit);

tstart = tic;

A = data.A;
b = data.b;
n = size(A,2);

k = 0;
x = x0;
Ax = A*x;
F = loss_smv(x,data) + nsm_funv(x,model);
funv_array(1) = F;
grad = loss_smg(x,data);
resi = norm(x - prox_oper_vec(x-grad, model,1),2);
resi_array(1) = resi;

while k<maxit
    
    if mod(k,dis_freq)==0
        fprintf('This is iteration %d of CGD, and residual = %f\n', k, resi);
    end
    
    if resi<eps
        fprintf('CGD achieves required accuracy\n');
        break;
    end

% ==========================================================
% % This is the Matlab implementation, which is much slower than C source
% % Mex file because of the for loop.
%
%     for kk=1:n
%         j = kk;   %  j = randi(n);
%         eAx = exp(Ax);
%         gj = (eAx./(1+eAx) - b)'*A(:,j)/m;
%         hj = sum(eAx./((1+eAx).^2).*(A(:,j).^2))/m;
%         yj = x(j) - gj/hj;
%         if yj > reg/hj
%             xjn = yj - reg/hj;
%         elseif yj < -reg/hj
%             xjn = yj + reg/hj;
%         else
%             xjn = 0;
%         end
%         
%         % use xjn-x(j) as the descent direction and perform line search
%         ss = 1;
%         d = xjn - x(j);
%         xt = x;
%         xt(j) = x(j) + ss*d;
%         Axt = Ax + A(:,j)*(xt(j) - x(j));
%         Fxt = loss_smv(xt,data) + nsm_funv(xt,model);
%         while Fxt - F > 0.5*ss*(gj*d + abs(xt(j)) - abs(x(j)))
%             ss = 0.25*ss;
%             xt(j) = x(j) + ss*d;
%             Axt = Ax + A(:,j)*(xt(j) - x(j));
%             Fxt = loss_smv(xt,data) + nsm_funv(xt,model);
%         end
%         x(j) = xjn;
%         Ax = Axt;
%     end
%     F = Fxt;
% ==========================================================


    [x,Ax] = rcordmin(x,A,b,Ax,reg); % perform n coordinate updates using Mex-file.

    
    k = k+1;
    funv_array(k+1) = loss_smv(x,data) + nsm_funv(x,model);
    grad = loss_smg(x,data);
    resi = norm(x - prox_oper_vec(x-grad,model,1),2);
    resi_array(k+1) = resi;
    
end

num_iter = k;
xopt = x;
cput = toc(tstart);

fprintf('CGD is finished\n\n');
        
end