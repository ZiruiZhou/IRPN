function output = prox_oper_vec( v, model, ss )

% This function evaluates the proximal operator prox_h(v)
% Note: ss is the so-called threshold, and it is allowed to be a vector.

if length(ss)==1
    rg = (ss*model.regpara)*ones(length(v),1);
elseif length(ss)==length(v)
    rg = ss*model.regpara;
else
    error('prox_oper_vec: lengh(ss) must be 1 or equal to length(v)');
end

pn = model.penalty;

switch pn
    case 'ell1'
        d = abs(v);
        d(d>=rg) = rg(d>=rg);
        add = d.*sign(v);
        output = v - add;
    case 'ell2'
        if norm(v,2)<=rg
            output = zero(size(v));
        else
            output = (1 - rg/norm(v,2))*v;
        end
    otherwise
        error('prox_oper_vec: nonsmooth functions must be l1 or l2 norm function');
end


end

