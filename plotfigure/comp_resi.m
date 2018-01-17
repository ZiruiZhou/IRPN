function resi_array = comp_resi(x_array,data,model)

% pre-processing

xp = sum(abs(x_array),1);
ncol = sum(xp>0);
x_array = x_array(:,1:ncol);
resi_array = zeros(1,ncol);

for i=1:ncol
    x = x_array(:,i);
    grad = loss_smg(x,data);
    resi_array(i) = norm(x - prox_oper_vec(x-grad,model,1),2);
end


end

