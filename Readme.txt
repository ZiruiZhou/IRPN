Matlab codes for numerical experiments in the paper:


M.-C. Yue, Z. Zhou, A. M.-C. So:

"A Family of Inexact SQA Methods for Non-Smooth Convex Minimization with Provable Convergence Guarantees Based on the Luo-Tseng Error Bound Property".


- Programmer: Zirui Zhou, Simon Fraser University, "http://www.sfu.ca/~ziruiz/".

- Questions, comments, and suggestions about the codes are welcome. 

- Zirui Zhou,  ziruiz@sfu.ca


- Last updated: Jan 15, 2018.


===============================================================================================
Installation:

1. To run the codes properly, users have to first download 'liblinear-2.20' and the three data sets 'colon-cancer', 'rcv1', 'news20' from LIBSVM website.

    - Download link for liblinear-2.20: http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/liblinear.cgi?+http://www.csie.ntu.edu.tw/~cjlin/liblinear+zip

    - Download link for these data sets:
        - 'colon-cancer': https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/colon-cancer.bz2
        - 'rcv1'        : https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2
        - 'news20'      : https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2

2. After downloading 'liblinear-2.20' and the three data sets:
    - 'liblinear-2.20' has to be unzipped and should be put in the main directory.
    - The three datasets should be unzipped and put in the 'Datasets' directory.

3. To finish installation, run 'setup.m' in the main directory.



================================================================================================

This directory contains the following Matlab source codes and folders:


Datasets    	- Folder. 		It is used to locate the data sets: 'colon-cancer', 'rcv1', and 'news20', all of which are downloadable from LIBSVM datasets: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/


functions   	- Folder. 		It contains some matlab functions for computing gradient, function value, proximal operators, etc.


liblinear-2.20 	- Folder. 		It contains the liblinear-2.20 software. We use the function 'libsvmread' in the repository "./liblinear-2.20/matlab/" for importing data sets.


plotfigure  	- Folder. 		It contains matlab files for generating the figures in our paper that show the convergence vs # of iterations (CPU time).


alg_cgd.m       - Matlab function. 	Apply CGD method for solving l1-regularized logistic regression.


alg_fista.m     - Matlab function. 	Apply FISTA for solving l1-regularized logistic regression.


alg_nglmnet.m	- Matlab function. 	Apply newGLMNET for solving l1-regularized logistic regression.
	

alg_nmt.m       - Matlab function. 	Apply SpaRSA (nonmonotone PGM) for solving l1-regularized logistic regression.


alg_rpn.m       - Matlab function. 	Apply IRPN for solving l1-regularized logistic regression.


setup.m         - Matlab function. 	Build all the necessary C codes.


rcordmin.c      - C code. 		It is used as a subroutine of alg_cgd.m. It runs n coordinate updates of CGD.


ripn_cordmin.c	- C code. 		It is used as a subroutine of alg_rpn.m. It runs n coordinate updates of coordinate descent method.


main_test.m	- Matlab script file. 	Perform the numerical test.


=====================================================================================================


