% This script plots the Figures in our paper (Figure 2), showing the
% convergence of r(x^k) against number of iterations and CPU time.

clear;

addpath '..'

dataset = 'rcv1.binary';
[resi_array0,tt0,resi0,ct0,resi_array05,tt05,resi05,ct05,resi_array1,tt1,resi1,ct1] = runirpn(dataset);


figure;
% plot convergence against outer iterations
semilogy(1:ct0,resi0,'-');
hold on
semilogy(1:ct05,resi05,'-.');
semilogy(1:ct1,resi1,'--');
legend('\rho=0','\rho=0.5','\rho=1');
xlabel('Number of Outer Iterations');
ylabel('log(r(x))');
title(dataset);

figure;
% plot convergence against time
semilogy(tt0,resi_array0,'-');
hold on
semilogy(tt05,resi_array05,'-.');
semilogy(tt1,resi_array1,'--');
legend('\rho=0','\rho=0.5','\rho=1');
xlabel('CPU Time (in seconds)');
ylabel('log(r(x))');
title(dataset);





