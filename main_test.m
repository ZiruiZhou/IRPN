% Notes: run setup.m before this script

clear;

fid = fopen('l1_logistic.txt','w');     % we write all results into this txt file

datasets = {'colon-cancer','rcv1.binary','news20.binary'}; % test datasets 


dataset = datasets{3};      % choose a dataset for test


% Input the data
datapath = strcat('./Datasets/', dataset);
[class_label, feature_matrix] = libsvmread(datapath);



% pre-process the data sets
switch dataset
    case 'colon-cancer'
        class_label((class_label==-1)) = 0; % make -1 labels be 0  
        m = size(feature_matrix,1);
        for i=1:m
            % scale the feature_matrix such that each row has norm 1
            feature_matrix(i,:) = feature_matrix(i,:)/norm(feature_matrix(i,:));    
        end
    case 'rcv1.binary'
        class_label((class_label==-1)) = 0; % make -1 labels be 0
    case 'news20.binary'
        class_label((class_label==-1)) = 0; % make -1 labels be 0
end



% form input data
data.A = feature_matrix;
data.b = class_label;
data.name = dataset;
[m,n] = size(data.A);
nnzd = nnz(data.A);


model.loss = 'logistic';
model.penalty = 'ell1';
model.regpara = 0.0005;
model.eps = 1e-4;

% a list of test algorithms
alg = {'fista_r','SpaRSA','CGD','newGLMNET', 'IRPN_0','IRPN_0.5', 'IRPN_1'};

% generate an initial point
x0 = 10*randn(n,1);

% line search parameters
opts.beta = 0.25;   
opts.sigma = 0.25; 

fprintf(fid, 'Numerical Test on Data Set %s.\n', dataset);
fprintf(fid, 'm = %d \t n = %d \t nnz = %d \t lambda = %1.1e \t eps = %1.1e \n', m,n,nnzd,model.regpara,model.eps);
fprintf(fid,'===============================================================\n');
fprintf(fid,'   Algorithm  | o_iter | in_iter | CPU time |  Obj. Value  \n');
fprintf(fid,'--------------------------------------------------------------\n');

for algi = 1:length(alg)
    algtest = alg{algi};
    switch algtest
        case 'fista_r'
            opts.maxit = 10000;
            opts.adp = 1;
            [funv, x, resi, numit, cput, Lipt] = alg_fista(x0, data, model, opts);
        case 'fista'
            opts.maxit = 100000;
            opts.adp = 0;
            [funv, x, resi, numit, cput, Lipt] = alg_fista(x0, data, model, opts);
        case 'SpaRSA'
            opts.maxit = 10000;
            opts.c = 1e-4;
            opts.M = 5;
            opts.ssu = 1e+8;
            opts.ssl = 1e-8;
            [funv, x, resi, numit, cput, ssa] = alg_nmt(x0, data, model, opts);
        case 'CGD'
            opts.maxit = 500;
            [funv, x, resi, numit, cput] = alg_cgd(x0, data, model, opts);
        case 'newGLMNET'
            opts.maxit = 100;
            opts.maxitsub = 100;
            opts.mu = 1e-12;
            [funv, x, resi, numit, cput, ssa] = alg_nglmnet(x0, data, model, opts);
        case 'IRPN_0'
            opts.maxit = 50;
            opts.maxitsub = 50;
            opts.mu = 1e-6;
            opts.eta = 0.5;
            opts.rho = 0;    
            [funv, x, resi, numit, cput, ssa] = alg_rpn(x0, data, model, opts);
        case 'IRPN_0.5'
            opts.maxit = 50;
            opts.maxitsub = 100;
            opts.mu = 1e-6;
            opts.eta = 0.5;
            opts.rho = 0.5;    
            [funv, x, resi, numit, cput, ssa] = alg_rpn(x0, data, model, opts);
        case 'IRPN_1'
            opts.maxit = 50;
            opts.maxitsub = 100;
            opts.mu = 1e-6;
            opts.eta = 0.5;
            opts.rho = 1;    
            [funv, x, resi, numit, cput, ssa] = alg_rpn(x0, data, model, opts);

        otherwise
            error('At least one of the following algorithms should be selected: fista_r, SpaRSA, CGD, newGLMNET,  IRPN_0, IRPN_0.5, IRPN_1');
    end
    fprintf(fid,'%12s: \t %4d \t  %4d \t   %3.2f \t   %2.6f\n', algtest, sum(numit>0), sum(numit), cput, min(funv(funv>0))); 
end


fclose(fid);


    