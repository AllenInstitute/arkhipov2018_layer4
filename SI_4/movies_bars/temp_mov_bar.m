clear variables; 

slen = 2500;   %temporal length of bar movie in ms.
nkt = 500; % padding time (in ms) for gray screen before the bar movie starts.
nxp = 192; nyp = 96;

%specify bar width and start and end points
bw = 6; x0 = 50; y0 = 0; 

flag_hor = 0;    % 1 is hor dirn of motion; otherwise vertical
flag_bl = 0;     % 0 is white bar; otherwise black

mov = zeros(nxp*nyp,slen);   %initialize

vel = 50.0;  %set speed of bar

out_name = 'Wbar_v50pixps_vert.mat';

%Make movie
for ii = 1:slen
    tmp_mov = zeros(nxp, nyp);
    if flag_hor == 1
        ori = 0;
        velx = vel*cos(ori); vely = vel*sin(ori);
        bxctr = mod((x0 + 10^-3*velx*ii),nxp);
        byctr = nyp/2;
        xpos = round(bxctr-bw/2):1:round(bxctr+bw/2);
        if flag_bl == 0
            tmp_mov(xpos(xpos>0),:) = 1;
        else
            tmp_mov(xpos(xpos>0),:) = -1;
        end
            
    else
        ori = pi/2;
        velx = vel*cos(ori); vely = vel*sin(ori);
        bxctr = nxp/2;
        byctr = mod((y0 + 10^-3*vely*ii),nyp);
        ypos = round(byctr-bw/2):1:round(byctr+bw/2);
        if flag_bl==0
            tmp_mov(:,ypos(ypos>0)) = 1;
        else
           tmp_mov(:,ypos(ypos>0)) = -1; 
        end
    end
    
    tmp_mov = tmp_mov(1:nxp,1:nyp); %truncate movie to correct dims in case of rounding errors
    
    for i = 1:nxp
        for j = 1:nyp
            mov((j-1)*nxp + i,ii) = tmp_mov(i, j);
        end
    end
end

gray = zeros(nxp*nyp,nkt);          %gray screen of specified length
mov = cat(2,gray,mov);      %do padding with movie

save(out_name,'mov');   %save

