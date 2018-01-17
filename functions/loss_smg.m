function smg = loss_smg(x, data)

% This function returns the gradient of the smooth loss function.


A = data.A;
b = data.b;
m = length(b);

Ax = A*x;
eAx = exp(Ax);
smg = A'*(eAx./(1+eAx) - b)/m;

end

