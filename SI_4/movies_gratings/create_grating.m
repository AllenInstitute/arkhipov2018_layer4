%Code to construct grating movies for every combination of (sf,tf,ori).
%Code generates 3000 frames for a given combination (3s long stim)
%%Constructed assuming sf is in cyc/deg and tf is in Hz;

clear variables;

%Load a file containing all grating conditions.
%----------------------------
load ('ORI_data.mat','VisLog');
bgsweeptable = VisLog.bgsweeptable;

% Choose the size of the screen in a linear angle approximation
% (using "lindegs" instead of normal degrees).
%----------------------------------------------------------------

size_used = 120; %half-angle size in lindegs
size_used_rads = size_used*pi/180; %convert lindegs to rads for use in grating formula

% Choose the number of points determining the resolution.  It seems proper
% to choose it in such a way that the spatial periods of most gratings
% would be respresented by even number of points (e.g., sf = 0.4 cpd
% corresponds to the period of 2.5 degrees; if we want to represent that by
% two points, we need (size_used / (2.5 degrees)) points.
N_p_1D = 192; %384; %We can use different resolutions for different sfs.

N_x_window = N_p_1D;
N_y_window = N_p_1D / 2;

delta_x = (N_p_1D - N_x_window) / 2;
if delta_x < 0
    delta_x = 0;
end
delta_y = (N_p_1D - N_y_window) / 2;
if delta_y < 0
    delta_y = 0;
end

%Time parameters for generating movie for each condition
%-------------------------------------------------------
dt = 1*10^-3;
t = 1*10^-3:dt:3; %time of stim in (s)
stim_time = ones(1,length(t));

%Generate a grid of (x,y) points on which the grating texture will be
%generated. This grid is generated on the whole virtual window. Finally
%a window corresponding to the actual screen size is 'cut out' of this
%virtual window when saving the grating movie. 
%----------------------------------------------------------------------

[x y] = meshgrid(linspace(-size_used_rads, size_used_rads, N_p_1D)); 
phi = pi/2;  %arbitrary phase factor to line up first frame with psycopy

mov = [];
tic

grating_each_sweep_k1_sqr_fine_full = cell(1,length(bgsweeptable));
f_m = fopen('res_192_metadata.txt','w');
for kk = 1:length(bgsweeptable)
    kk
    savename = strcat('res_192/','grating_',num2str(kk),'.mat'); %path to save mat file
    ori = bgsweeptable(kk,1);
    sf =  bgsweeptable(kk,2)*stim_time;
    tf =  bgsweeptable(kk,3)*stim_time;
    % Convert spatial frequency from psychopy "angles" to real angles, so that we can
    % keep track of that.  Assume that conversions in x and y are the same,
    % and do the actual conversion only using size in the x dimension.
    contrast = 0.8*stim_time;
    fprintf(f_m, '%s %f %f %f %f\n', savename, ori, sf(1), tf(1), contrast(1));
    
    % We can use the if statement below to produce movies only for
    % specific conditions; for example, we can write movies with higher
    % resolution only for specific SFs.
    if true
        for i = 1:length(t)
            ori_rad = ori * pi / 180.0;
            temp = contrast(i)*sign(sin(360*sf(i)*(y * cos(ori_rad) + x * sin(ori_rad)) + phi - 2*pi*tf(i)*(t(i)-t(1))));
            temp = temp(delta_x+1:end-delta_x,delta_y+1:end-delta_y); %%HARD_CODED for k = 0.5
            mov(:,i) = temp(:);
        end
    save(savename,'mov');   % Save each stim condition
    fprintf('stimc %d done',kk)
    end
end
fclose(f_m);
toc

