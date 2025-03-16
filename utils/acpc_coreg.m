%%acpc_coreg.m
% Batch script for AC-PC reorientation
% This script tries to set AC-PC with 2 steps.
% 1. Set origin to center (utilizing a script by F. Yamashita)
% 2. Coregistration of the image to icbm152.nii under spm/toolbox/DARTEL
% 
% K. Nemoto 22/May/2017

%% Initialize batch
spm_jobman('initcfg');
matlabbatch = {};

%% Select images
imglist = spm_select(Inf, 'image', 'Choose MRI you want to set AC-PC');

%% Set the origin to the center of the image
% This part is written by Fumio Yamashita.
for i = 1:size(imglist, 1)
    file = deblank(imglist(i, :));
    st.vol = spm_vol(file);
    vs = st.vol.mat \ eye(4);
    vs(1:3, 4) = (st.vol.dim + 1) / 2;
    spm_get_space(st.vol.fname, inv(vs));
end

%% Prepare the SPM window
% Interactive window (bottom-left) to show the progress, 
% and graphics window (right) to show the result of coregistration 

%spm('CreateMenuWin','on'); %Comment out if you want the top-left window.
spm('CreateIntWin', 'on');
spm_figure('Create', 'Graphics', 'Graphics', 'on');

%% Coregister images with icbm152.nii under spm12/toolbox/DARTEL
for i = 1:size(imglist, 1)
    matlabbatch{i}.spm.spatial.coreg.estimate.ref = {fullfile(spm('dir'), 'toolbox', 'DARTEL', 'icbm152.nii,1')};
    matlabbatch{i}.spm.spatial.coreg.estimate.source = {deblank(imglist(i, :))};
    matlabbatch{i}.spm.spatial.coreg.estimate.other = {''};
    matlabbatch{i}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
    matlabbatch{i}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];
    matlabbatch{i}.spm.spatial.coreg.estimate.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
    matlabbatch{i}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];
end

%% Run batch
%spm_jobman('interactive',matlabbatch);
spm_jobman('run', matlabbatch);

% Set JSON file path
output_path = '/mnt/hdd1/users/21011961_seok/output_matrix.json';

% Get key name input from user
key_name = input('Enter the key name to save: ', 's');

% Get current Figure handle
figHandle = gcf; % Active Figure handle

% Find axes objects
axesHandles = findobj(figHandle, 'Type', 'axes');

% Search for text objects and extract numbers:
disp('Searching for text objects and extracting numbers:');
textData = strings(1, 3); % Store in order: X1, Y1, Z1
for i = 1:length(axesHandles)
    % Find all text objects within the axes
    textHandles = findobj(axesHandles(i), 'Type', 'text');
    
    for j = 1:length(textHandles)
        % Get the String value of the text object
        textString = get(textHandles(j), 'String');
        
        % Save according to keywords X1, Y1, Z1
        if contains(textString, 'Z1')
            textData(3) = textString; % Z1 is stored in 3rd position
            fprintf('Z1 info: %s\n', textString);
        elseif contains(textString, 'Y1')
            textData(2) = textString; % Y1 is stored in 2nd position
            fprintf('Y1 info: %s\n', textString);
        elseif contains(textString, 'X1')
            textData(1) = textString; % X1 is stored in 1st position
            fprintf('X1 info: %s\n', textString);
        end
    end
end

% Create matrix from numbers extracted from the text
matrix_data = zeros(4, 4); % Initialize 4x4 matrix
for i = 1:3
    % Extract numbers from each text data
    parsedValues = sscanf(textData(i), '%*s = %f*X %f*Y %f*Z %f');
    matrix_data(i, :) = [parsedValues(:)'];
end

% Add the last row [0, 0, 0, 1.0]
matrix_data(4, :) = [0, 0, 0, 1.0];

% Read existing JSON file (if exists)
if isfile(output_path)
    % Read the existing JSON file
    fid = fopen(output_path, 'r');
    raw = fread(fid, inf, 'char');
    fclose(fid);
    existing_data = jsondecode(char(raw')); % Decode existing data
else
    % Initialize an empty structure if JSON file does not exist
    existing_data = struct();
end

% Add new data to the existing data
existing_data.(key_name) = matrix_data;

% Save to JSON file
json_text = jsonencode(existing_data); % Convert to JSON format
fid = fopen(output_path, 'w');
if fid == -1
    error('Cannot open file: %s', output_path);
end
fwrite(fid, json_text, 'char'); % Write JSON data to file
fclose(fid);

disp(['JSON file saved: ', output_path]);
