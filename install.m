% Compile libraries and download network
setup_paths();
[home_dir, name, ext] = fileparts(mfilename('fullpath'));

warning('ON', 'SASR:install')

% mtimesx
if exist('external_libs/mtimesx', 'dir') == 7
    cd external_libs/mtimesx
    mtimesx_build;
    cd(home_dir)
else
    error('SASR:install', 'Mtimesx not found.')
end

% PDollar toolbox
if exist('external_libs/pdollar_toolbox/external', 'dir') == 7
    cd external_libs/pdollar_toolbox/external
    toolboxCompile;
    cd(home_dir)
else
    warning('SASR:install', 'PDollars toolbox not found. Clone this submodule if you want to use HOG features. Skipping for now.')
end

% matconvnet
if exist('external_libs/matconvnet/matlab', 'dir') == 7
    cd external_libs/matconvnet/matlab
    try
        disp('Trying to compile MatConvNet with GPU support')
        vl_compilenn('enableGpu', true)
%         vl_compilenn('enableGpu',true,...
%             'cudaRoot','C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0',...% path to your cuda
%             'cudaMethod','nvcc',...
%             'enableCudnn',true,...    % if you don't want to use cudnn, set it to 'false'
%             'cudnnRoot','C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0'); % path to your cudnn
        disp('GPU support completed')
    catch err
        warning('SASR:install', 'Could not compile MatConvNet with GPU support. Compiling for only CPU instead.\nVisit http://www.vlfeat.org/matconvnet/install/ for instructions of how to compile MatConvNet.\nNote: remember to move the mex-files after re-compiling.');
        vl_compilenn;
    end
    status = movefile('mex/vl_*.mex*');
    cd(home_dir)
    
    % donwload network
    cd feature_extraction
    mkdir networks
    cd networks
    if ~(exist('imagenet-vgg-m-2048.mat', 'file') == 2)
        disp('Downloading the network "imagenet-vgg-m-2048.mat" from "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m-2048.mat"...')
        urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m-2048.mat', 'imagenet-vgg-m-2048.mat')
        disp('Done!')
    end
    cd(home_dir)
else
    warning('SASR:install', 'Matconvnet not found. Clone this submodule if you want to use CNN features. Skipping for now.')
end