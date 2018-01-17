function funv = loss_smv(x, data)

% This function evaluates the function value of the smooth loss function

A = data.A;
b = data.b;
m = length(b);


Ax = A*x;

funv = (sum(log(1+exp(Ax))) - dot(b,Ax))/m;

end

