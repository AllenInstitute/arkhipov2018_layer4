%Code to construct movies of full-screen flashes.
%Code generates 3000 frames (3s long)

clear variables;


%Initialize parameters
%---------------------
nxp = 16;
nyp = 8;


%Parameters from logfile complying with psychopy's virtual window
%----------------------------------------------------------------

size_used = 175; %half-angle size in degrees (actual size was 350 x 350 deg virtual window)
size_used_rads = 175*pi/180; %convert deg to rads for use in grating formula

%Time parameters for generating movie for each condition
%-------------------------------------------------------
dt = 1*10^-3;
t = 1*10^-3:dt:3; %time of stim in (s)
stim_time = ones(1,length(t));

%Generate a grid of (x,y) points. This grid is generated on the whole
%virtual window. Finally a window corresponding to the actual screen
%size is 'cut out' of this virtual window when saving the movie. 
%----------------------------------------------------------------------

a = 2;      %resolution needed
k = a*0.5;  % Chosen so that MATLAB does not crash when saving
delta = a*24;% Chosen ad-hoc to get the eventual cutout window close to 254x142 degrees (actual screen size) ()
[x y] = meshgrid(linspace(-size_used_rads, size_used_rads, k*nxp*nyp+delta)); 

mov = [];
tic

kk = 2 %1
savename = strcat('flash_',num2str(kk),'.mat'); %path to save mat file
for i = 1:length(t)
    scr_val = 0.0;
    if i > 600 %1000
        scr_val = 1.0;
    end
    if i > 650 %2000
        scr_val = 0.0;
    end
    temp = scr_val + 0.0 * (x + y); % Using x and y here to create an array.
    temp = temp(delta/2+1:end-delta/2,delta+a*4+1:end-(delta+a*4)); %%HARD_CODED for k = 0.5
    mov(:,i) = temp(:);
end
save(savename,'mov');   % Save to file.
toc

