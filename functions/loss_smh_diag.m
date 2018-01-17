function smh_dg = loss_smh_diag(x,data)

% This function returns a diagonal component matrix D such that the hessian
% matrix of the 'logistic loss' function H = A'*D*A, where A = data.A

% Note: the output smh_dg is of MATLAB 'sparse matrix' type.


A = data.A;
n = size(A,1);



Ax = A*x;
eAx = exp(Ax);
smh_dg = ((eAx)./((1+eAx).^2))/n;
% ir = 1:n;
% jc = 1:n;
% smh_dg = sparse(ir,jc,P,n,n);


end

