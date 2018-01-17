function [resi_array0,tt0,resi0,ct0,resi_array05,tt05,resi05,ct05,resi_array1,tt1,resi1,ct1] = runirpn(dataset)

% Input the data
datapath = strcat('../Datasets/', dataset);
[class_label, feature_matrix] = libsvmread(datapath);
fprintf('Dataset has been successfully read.\n');


switch dataset
    case 'colon-cancer'
        class_label((class_label==-1)) = 0; % make -1 labels be 0  
        m = size(feature_matrix,1);
        for i=1:m
            feature_matrix(i,:) = feature_matrix(i,:)/norm(feature_matrix(i,:));
        end
    case 'rcv1.binary'
        class_label((class_label==-1)) = 0; % make -1 labels be 0
    case 'news20.binary'
        class_label((class_label==-1)) = 0; % make -1 labels be 0
end
 



% form input data
n = size(feature_matrix,2);
data.A = feature_matrix;
data.b = class_label;
data.name = dataset;

model.loss = 'logistic';
model.penalty = 'ell1';
model.regpara = 0.0005;
model.eps = 1e-8;

% choose an algorithm to solve problem
alg = {'rpn_cd_0','rpn_cd_0.5','rpn_cd_1'};

% generate an initial point
x0 = 10*randn(n,1);

% line search parameters
opts.beta = 0.25;   
opts.sigma = 0.25; 


for algi = 1:length(alg)
    algtest = alg{algi};
    switch algtest
        case 'rpn_cd_0'
            opts.maxit = 50;
            opts.maxitsub = 50;
            opts.mu = 1e-6;
            opts.eta = 0.5;
            opts.rho = 0;    
            [~,x_array0,tt0,resi0,~,~,~] = alg_rpn_tt(x0, data, model, opts);
        case 'rpn_cd_0.5'
            opts.maxit = 50;
            opts.maxitsub = 100;
            opts.mu = 1e-6;
            opts.eta = 0.5;
            opts.rho = 0.5;    
            [~,x_array05,tt05,resi05,~,~,~] = alg_rpn_tt(x0, data, model, opts);
        case 'rpn_cd_1'
            opts.maxit = 50;
            opts.maxitsub = 100;
            opts.mu = 1e-6;
            opts.eta = 0.5;
            opts.rho = 1;    
            [~,x_array1,tt1,resi1,~,~,~] = alg_rpn_tt(x0, data, model, opts);

        otherwise
            error('At least one of the following algorithms should be selected:ista, fista, rpn_cd, rpn_ssn.');
    end
    
end

% pre-processing resi_array's
resi_array0 = comp_resi(x_array0,data,model);
resi_array05 = comp_resi(x_array05,data,model);
resi_array1 = comp_resi(x_array1,data,model);

tt1 = tt1(1:length(resi_array1));
tt0 = tt0(1:length(resi_array0));
tt05 = tt05(1:length(resi_array05));

% pre-processing resi's
ct0 = sum(resi0>0);
resi0 = resi0(1:ct0);
ct05 = sum(resi05>0);
resi05 = resi05(1:ct05);
ct1 = sum(resi1>0);
resi1 = resi1(1:ct1);


    
end

