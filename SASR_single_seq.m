
clc;
clear;
close all;

anno_path = 'seq\anno\';
img_path = 'seq\data_seq\';
setup_paths();


video_name = choose_video_UAV(anno_path);
database_folder = img_path;
seq = load_video_info_UAV123(video_name, database_folder, anno_path, 'UAV123_10fps');

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