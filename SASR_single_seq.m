% Here is a demo for SASR "Surrounding-Aware Correlation Filter for UAV Tracking  with Selective Spatial Regularization" 

clc;
clear;
close all;

% setup path for video sequences and annotation
anno_path = 'seq\anno\';
img_path = 'seq\data_seq\';
setup_paths();

% load an video and then load information for tracking
% the sequence here comes from 'UAV123_10fps'
video_name = choose_video_UAV(anno_path);
seq = load_video_info_UAV123(video_name, img_path, anno_path, 'UAV123_10fps');

% main function
result  =   run_SASR(seq,0,0);

% save results
results = cell(1,1);
results{1} = result;
results{1}.len = seq.len;
results{1}.startFrame = seq.st_frame;
results{1}.annoBegin = seq.st_frame;

% save results to specified folder
save_dir = '.\result\';
save([save_dir, seq.video_name, '_', 'SASR.mat'], 'results');

% plot precision figure
show_visualization =1;
precision_plot_save(results{1}.res, seq.ground_truth, seq.video_name, save_pic_dir, show_visualization);

close all;