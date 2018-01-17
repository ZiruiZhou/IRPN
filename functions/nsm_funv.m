function funv = nsm_funv(x, model)

% This function evaluates the function value of the nonsmooth regularizer

rg = model.regpara;
pn = model.penalty;

switch pn
    case 'ell1'
        funv = rg*norm(x,1);
    case 'ell2'
        funv = rg*norm(x,2);
    otherwise
        error('nsm_funv: nonsmooth regularizer must be l1 or l2 norm function');
end


end

