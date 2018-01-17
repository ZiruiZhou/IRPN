function setup()

% users have to download 'liblinear-2.20'
% and the three data sets 'colon-cancer', 'rcv1', 'news20' from LIBSVM
% website.
% 
% Download link for liblinear-2.20: http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/liblinear.cgi?+http://www.csie.ntu.edu.tw/~cjlin/liblinear+zip
% 
% Download link for these data sets:
%   - 'colon-cancer': https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/colon-cancer.bz2
%   - 'rcv1'        : https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2
%   - 'news20'      : https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2
% 
% - 'liblinear-2.20' should be put in the same directory as this script.
% - The three datasets should be put in the 'Datasets' directory.
% - The three datasets have to be unzipped.
% compile the Mex files 


%% Check if the data sets are ready
datasets = {'colon-cancer','rcv1.binary','news20.binary'}; % test datasets 

for i=1:length(datasets)
    dataset = datasets{i};
    datapath = strcat('./Datasets/', dataset);
    if ~exist(datapath, 'file')
        error('The test data set %s is not in the Datasets folder', dataset);
    end
end

%% Check if liblinear-2.20 is ready
libsvmreadpath = './liblinear-2.20/matlab';
if ~isdir(libsvmreadpath)
    error('Our experiments require the use of liblinear-2.20');
end


%% Add paths

addpath(libsvmreadpath);
addpath './functions'
addpath './Datasets'


%% Build C source files into Matlab Mex-files

cd './liblinear-2.20/matlab'
make

cd ../..

mex -largeArrayDims ripn_cordmin.c  % this is a C program for coordinate minimization
mex -largeArrayDims rcordmin.c

end

