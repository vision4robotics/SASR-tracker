function results = tracker(params)


%% Initialization
back_imp=params.back_imp;
offset_prop=params.offset_prop;
learning_rate=params.learning_rate;

% Get sequence info
[seq, im] = get_sequence_info(params.seq);
params = rmfield(params, 'seq');%remove params.seq from params
if isempty(im)
    seq.rect_position = [];
    [~, results] = get_sequence_results(seq);
    return;
end

% Init position
pos = seq.init_pos(:)';
target_sz = seq.init_sz;
params.init_sz = target_sz;

% Feature settings
features = params.t_features;

% Set default parameters
params = init_default_params(params);

% Global feature parameters
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end

global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;

% Define data types
if params.use_gpu
    params.data_type = zeros(1, 'single', 'gpuArray');
else
    params.data_type = zeros(1, 'single');
end
params.data_type_complex = complex(params.data_type);

global_fparams.data_type = params.data_type;

% Load learning parameters
admm_max_iterations = params.max_iterations;
init_penalty_factor = params.init_penalty_factor;
max_penalty_factor = params.max_penalty_factor;
penalty_scale_step = params.penalty_scale_step;

init_target_sz = target_sz;

% Check if color image
if size(im,3) == 3
    if all( all( im(:,:,1) == im(: ,: ,2 ) ) )
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end

% Check if mexResize is available and show warning otherwise.
params.use_mexResize = true;
global_fparams.use_mexResize = true;

try
    [~] = mexResize(ones(5,5,3,'uint8'), [3 3], 'auto');
catch err
    params.use_mexResize = false;
    global_fparams.use_mexResize = false;
end

% Calculate search area and initial scale factor
search_area = prod(init_target_sz * params.search_area_scale);
if search_area > params.max_image_sample_size
    currentScaleFactor = sqrt(search_area / params.max_image_sample_size);
elseif search_area < params.min_image_sample_size
    currentScaleFactor = sqrt(search_area / params.min_image_sample_size);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        img_sample_sz = floor(base_target_sz * params.search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        img_sample_sz = repmat(sqrt(prod(base_target_sz * params.search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        img_sample_sz = base_target_sz + sqrt(prod(base_target_sz * params.search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    case 'custom'
        img_sample_sz = [base_target_sz(1)*2 base_target_sz(2)*2];
end

[features, global_fparams, feature_info] = init_features(features, global_fparams, is_color_image, img_sample_sz, 'exact');

% Set feature info
img_support_sz = feature_info.img_support_sz;
feature_sz = unique(feature_info.data_sz, 'rows', 'stable');
feature_cell_sz = unique(feature_info.min_cell_size, 'rows', 'stable');
num_feature_blocks = size(feature_sz, 1);

% Get feature specific parameters
feature_extract_info = get_feature_extract_info(features);

% Size of the extracted feature maps
feature_sz_cell = mat2cell(feature_sz, ones(1,num_feature_blocks), 2);
filter_sz = feature_sz;
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% The size of the label function DFT. Equal to the maximum filter size
[output_sz, k1] = max(filter_sz, [], 1);
k1 = k1(1);

% Get the remaining block indices
block_inds = 1:num_feature_blocks;
block_inds(k1) = [];


% Construct the Gaussian label function
yf = cell(numel(num_feature_blocks), 1);
for i = 1:num_feature_blocks
    sz = filter_sz_cell{i};
    output_sigma = sqrt(prod(floor(base_target_sz/feature_cell_sz(i)))) * params.output_sigma_factor;
    rg           = circshift(-floor((sz(1)-1)/2):ceil((sz(1)-1)/2), [0 -floor((sz(1)-1)/2)]);
    cg           = circshift(-floor((sz(2)-1)/2):ceil((sz(2)-1)/2), [0 -floor((sz(2)-1)/2)]);
    [rs, cs]     = ndgrid(rg,cg);
    y            = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
    yf{i}           = fft2(y);
end

% Compute the cosine windows
cos_window = cellfun(@(sz) hann(sz(1))*hann(sz(2))', feature_sz_cell, 'uniformoutput', false);

% Define spatial regularization windows
reg_window = cell(num_feature_blocks, 1);
for i = 1:num_feature_blocks
    reg_scale = floor(base_target_sz/params.feature_downsample_ratio(i));
    use_sz = filter_sz_cell{i};
    wrg = round( ( 1:use_sz(1) ) - use_sz(1)/2 );
    wcg = round( ( 1:use_sz(2) ) - use_sz(2)/2 );
    [wrs,wcs]=ndgrid(wrg , wcg);
    reg_window_temp = params.eta * ( (wrs/base_target_sz(1)) .^ 2 + (wcs/base_target_sz(2)) .^ 2 ) ; %elipse shaped regularization
    critical_value = floor( use_sz/2 + params.regularizer_size * reg_scale/2 );
    reg_window_temp = reg_window_temp - reg_window_temp(critical_value(1) , critical_value(2)) ;
    reg_window_temp(reg_window_temp<0)=0;
    reg_window_temp =  reg_window_temp + params.reg_window_min;
    reg_window_temp(reg_window_temp > params.reg_window_max) = params.reg_window_max;
    reg_window{i}=reg_window_temp;
end

% Pre-computes the grid that is used for socre optimization
ky = circshift(-floor((filter_sz_cell{1}(1) - 1)/2) : ceil((filter_sz_cell{1}(1) - 1)/2), [1, -floor((filter_sz_cell{1}(1) - 1)/2)]);
kx = circshift(-floor((filter_sz_cell{1}(2) - 1)/2) : ceil((filter_sz_cell{1}(2) - 1)/2), [1, -floor((filter_sz_cell{1}(2) - 1)/2)])';
newton_iterations = params.newton_iterations;

% Use the translation filter to estimate the scale
nScales = params.number_of_scales;
scale_step = params.scale_step;
scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
scaleFactors = scale_step .^ scale_exp;

if nScales > 0
    %force reasonable scale changes
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

seq.time = 0;

% Define the learning variables
cf_f = cell(num_feature_blocks, 1);

%pre-allocate memory
scores_fs_feat = cell(1,1,num_feature_blocks);
model_xf=cell(1,numel(num_feature_blocks));
while true
    % Read image
    if seq.frame > 0
        [seq, im] = get_sequence_frame(seq);
        if isempty(im)
            break;
        end
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1);
        end
    else
        seq.frame = 1;
    end
    
    tic();
    
    %% Target localization step
    
    % Do not estimate translation and scaling on the first frame, since we
    % just want to initialize the tracker there
    if seq.frame > 1
        
        old_pos = inf(size(pos));
        iter = 1;
        
        %translation search
        while iter <= params.refinement_iterations && any(old_pos ~= pos)
            % Extract features at multiple resolutions
            sample_pos = round(pos);
            sample_scale = currentScaleFactor*scaleFactors;
            xt = extract_features(im, sample_pos, sample_scale, features, global_fparams, feature_extract_info);
            
            % Do windowing of features
            xtw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt, cos_window, 'uniformoutput', false);
            
            % Compute the fourier series
            xtf = cellfun(@fft2, xtw, 'uniformoutput', false);
            
            % Compute convolution for each feature block in the Fourier domain
            % and the sum over all blocks.
            scores_fs_feat{k1} = gather(sum(bsxfun(@times, conj(cf_f{k1}), xtf{k1}), 3) );
            scores_fs_sum = scores_fs_feat{k1};
            for k = block_inds
                scores_fs_feat{k} = gather(sum(bsxfun(@times, conj(cf_f{k}), xtf{k}), 3));
                scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz);
                scores_fs_sum = scores_fs_sum +  scores_fs_feat{k};
            end
            
            % Also sum over all feature blocks.
            % Gives the fourier coefficients of the convolution response.
            scores_fs = permute(gather(scores_fs_sum), [1 2 4 3]);
            
            responsef_padded = resizeDFT2(scores_fs, output_sz);
            response = ifft2(responsef_padded, 'symmetric');
            [disp_row, disp_col, sind] = resp_newton(response, responsef_padded, newton_iterations, ky, kx, output_sz);
            
            % Compute the translation vector in pixel-coordinates and round
            % to the closest integer pixel.
            translation_vec = [disp_row, disp_col] .* (img_support_sz./output_sz) * currentScaleFactor * scaleFactors(sind);
            scale_change_factor = scaleFactors(sind);
            
            % update position
            old_pos = pos;
            pos = sample_pos + translation_vec;
            
            if params.clamp_position
                pos = max([1 1], min([size(im,1) size(im,2)], pos));
            end
            
            % Update the scale
            currentScaleFactor = currentScaleFactor * scale_change_factor;
            
            % Adjust to make sure we are not to large or to small
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
            
            iter = iter + 1;
        end
    end
    
    %% Model update step
    % extract image region for training sample
    
    offset_len=sqrt(prod(target_sz));
    if target_sz(2)>target_sz(1)
        asp_ratio=target_sz(2)/target_sz(1);
        offset_xf=offset_prop;
        offset_yf=offset_xf*sqrt(asp_ratio);
    else
        asp_ratio=target_sz(1)/target_sz(2);
        offset_yf=offset_prop;
        offset_xf=offset_yf*sqrt(asp_ratio);
    end
    offset = floor( [-offset_xf*offset_len 0 ;  0 -offset_yf*offset_len ;  offset_xf*offset_len 0 ;  0 offset_yf*offset_len] );
    
    
    sample_pos = round(pos);
    xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
    xl_u = extract_features(im, sample_pos+offset(1,:), currentScaleFactor, features, global_fparams, feature_extract_info);
    xl_l = extract_features(im, sample_pos+offset(2,:), currentScaleFactor, features, global_fparams, feature_extract_info);
    xl_d = extract_features(im, sample_pos+offset(3,:), currentScaleFactor, features, global_fparams, feature_extract_info);
    xl_r = extract_features(im, sample_pos+offset(4,:), currentScaleFactor, features, global_fparams, feature_extract_info);
    
    % do windowing of features
    xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
    xlw_u = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl_u, cos_window, 'uniformoutput', false);
    xlw_l = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl_l, cos_window, 'uniformoutput', false);
    xlw_d = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl_d, cos_window, 'uniformoutput', false);
    xlw_r = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl_r, cos_window, 'uniformoutput', false);
    
    % compute the fourier series
    xlf = cellfun(@fft2, xlw, 'uniformoutput', false);
    xlf_u = cellfun(@fft2, xlw_u, 'uniformoutput', false);
    xlf_l = cellfun(@fft2, xlw_l, 'uniformoutput', false);
    xlf_d = cellfun(@fft2, xlw_d, 'uniformoutput', false);
    xlf_r = cellfun(@fft2, xlw_r, 'uniformoutput', false);
    
    % train the CF model for each feature
    % hog and cn are actually concatenated, take them as one feature.
    
    for k = 1: numel(xlf)
        if (seq.frame == 1)
            model_xf{k} = xlf{k};
            extend_xf=cat(4,model_xf{k} , back_imp*xlf_u{k} , back_imp*xlf_l{k} , back_imp*xlf_d{k} , back_imp*xlf_r{k} );
        else
            model_xf{k} = ((1 - learning_rate) * model_xf{k}) + (learning_rate * xlf{k});
            extend_xf=cat(4,model_xf{k} , back_imp*xlf_u{k} , back_imp*xlf_l{k} , back_imp*xlf_d{k} , back_imp*xlf_r{k} );
        end
        
        % intialize the variables
        filter_f = single(zeros(size(model_xf{k})));
        g_f = filter_f;
        h_f = filter_f;
        gamma  = init_penalty_factor(k);
        gamma_max = max_penalty_factor(k);
        gamma_scale_step = penalty_scale_step(k);
        
        % use the GPU mode
        if params.use_gpu
            model_xf{k} = gpuArray(model_xf{k});
            filter_f = gpuArray(filter_f);
            g_f = gpuArray(g_f);
            h_f = gpuArray(h_f);
            reg_window{k} = gpuArray(reg_window{k});
            yf{k} = gpuArray(yf{k});
        end
        
        % pre-compute the variables
        T = prod(output_sz);
        S_xx = sum( sum( conj(extend_xf) .* extend_xf, 3) ,4);
        
        % solve via ADMM algorithm
        iter = 1;
        while (iter <= admm_max_iterations)
            
            % subproblem f
            B = S_xx + T * gamma;
            Sgx_f = sum(conj(extend_xf) .* g_f, 3);
            
            Shx_f = sum(conj(extend_xf) .* h_f, 3);
            
            filter_f = ((1/(T*(gamma)) * bsxfun(@times,  yf{k}, model_xf{k})) - ( (1/gamma) * h_f) + g_f) ...
                - ...minus
                bsxfun(@rdivide,...divided by B
                (...
                1/(T*(gamma)) * bsxfun(@times, model_xf{k}, (S_xx .*  yf{k})) ...
                - (1/(gamma))* sum(bsxfun(@times, extend_xf, Shx_f) , 4) ...
                +sum( bsxfun(@times, extend_xf, Sgx_f) ,4 )...
                ) , B);
            
            %   subproblem g
            g_f = fft2( argmin_g( reg_window{k}, gamma, real(ifft2(gamma * filter_f+ h_f)), g_f ) );
            
            %   update h
            h_f = h_f + (gamma * (filter_f - g_f));
            
            %   update gamma
            gamma = min(gamma_scale_step * gamma, gamma_max);
            
            iter = iter+1;
        end
        
        % save the trained filters
        cf_f{k} = filter_f;
    end
    
    % Update the target size (only used for computing output box)
    target_sz = base_target_sz * currentScaleFactor;
    
    %save position and calculate FPS
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    seq = report_tracking_result(seq, tracking_result);

    seq.time = seq.time + toc();
    %% Visualization
    if params.visualization == 1
        rect_position_vis = [pos( [2 , 1] ) - target_sz( [2 , 1] )/2 ,  target_sz( [2 , 1] )] ;
        im_to_show = double( im )/255 ;
        if size( im_to_show , 3 ) == 1
            im_to_show = repmat( im_to_show ,  [1 1 3] ) ;
        end
        
        if seq.frame == 1
            figure( 'Name' ,  'SASR' ) ;
            im_handle = imshow(im_to_show, 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));
            rect_handle=rectangle( 'Position' , rect_position_vis ,  'EdgeColor' , 'g' ,  'LineWidth' , 1 ) ;
            frame_handle = text(20, 30, ['Frame : ' int2str(seq.frame) '/' int2str( seq.len )], 'FontSize', 12, 'color', [0 1 1]);
            fps_handle=text( 20 ,  60 ,  [ 'FPS : ' num2str( 1/( seq.time/seq.frame ) ) ] ,  'color' ,  [0 1 1] ,  'fontsize' ,  12 ) ;
        else
            hold on;
            set(im_handle, 'CData', im_to_show)
            set(rect_handle, 'Position', rect_position_vis)
            set(frame_handle, 'string', ['Frame : ' int2str(seq.frame) '/' int2str( seq.len )]);
            set(fps_handle,'string',[ 'FPS : ' num2str( 1/( seq.time/seq.frame ) ) ]);
            resp_sz = round( img_sample_sz*currentScaleFactor*scaleFactors( sind ) ) ;
            xs = floor( old_pos( 2 ) ) + ( 1:resp_sz( 2 ) ) - floor( resp_sz( 2 )/2 ) ;
            ys = floor( old_pos( 1 ) ) + ( 1:resp_sz( 1 ) ) - floor( resp_sz( 1 )/2 ) ;
            if seq.frame==2 %response map
                resp_handle = imagesc( xs ,  ys ,  fftshift( response( : , : , sind ) ) ) ;  colormap hsv ;
                alpha( resp_handle ,  0.15 ) ;
            else
                set(resp_handle,'CData',fftshift( response( : , : , sind )),'XData',xs,'YData',ys)
            end
            drawnow
            hold off;
        end
        
    end
    %% output results
    [~, results] = get_sequence_results(seq);
    %     disp(['fps: ' num2str(results.fps)])
end
