%%
% Perform image processing and dictionary matching for Multimapping
% sequence to generate T1 and T2 maps of the heart.
% Using the following ECG-triggered pulse sequence across 10 cardiac cycles:
% ind  Rwave
%  1   |...Inv....ACQ...
%  2   |..........ACQ...
%  3   |..........ACQ...
%  4   |..........ACQ...
%  5   |...Inv....ACQ...
%  6   |..........ACQ...
%  7   |..........ACQ...
%  8   |......T2p1ACQ...
%  9   |......T2p2ACQ...
%  10  |......T2p3ACQ...
%
% 
%  Example images from one healthy volunteer are provided, including
%  necessary imaging parameters and subject-specific delays.
%  The dictionary matching is based on EPG simulations implemented by Dr
%  Shaihan Malik and have been modified for this work.
% 
% The scripts are implemented on Matlab R2020b using the Image processing and
% Parallel computing toolboxes


%% libraries needed for the EPG simulations
clear
load('example_data.mat')
addpath(genpath('libs'));

n_images = size(mag,3);
nr_startups = numel(SU_fa_list);
ref_im = 4; %image with largest Mz magnetization

%% Determine Mz polarity of signal

IM_PC_M = mag;
IM_PC_P = phase-phase(:,:,ref_im);
IM_phase_corrected = real(IM_PC_M.*exp(1i*(IM_PC_P)));

%% Create array of all pixel values to be used for dictionary matching

n_pixels = size(IM_phase_corrected,1)*size(IM_phase_corrected,2);

sig = shiftdim(IM_phase_corrected, 2);
sig = reshape(sig, [n_images, n_pixels]);

%% define RF npulses for dictionary (imaging flip angles will be defined later during B1 determination)

t2prep_rf = [90 180 90];
inv_rf = 160;
fa = info.FlipAngle;
n_ex = info.EchoTrainLength;

npulse = n_ex + nr_startups + 4; %maximum number of EPG entries for a cardiac cycle. 4 = no RF in T2prep + trigger delay

TR= info.RepetitionTime;

%% define phase for RF pulses
phi_bSSFP = RF_phase_cycle(npulse,'balanced');

for i = 1:n_images
    phi(i,:) = phi_bSSFP;
end

phi_t2prep = d2r([0 90 180]);

% T2prep are performed first in cardiac cyle 8, 9 & 10
phi(8,1:3) = phi_t2prep; 
phi(9,1:3) = phi_t2prep;
phi(10,1:3) = phi_t2prep;

%% Determine delays

%Delay_dur = RR interval minus all fixed durations

for i = 1:numel(RR_int_list)
    Delay_dur(i) = RR_int_list(i) - T2prep_dur - TFEPP_dur - TFEPP_delay_dur - TR*(n_ex + nr_startups);
end

% TNT_dur = duration from last RF pulse in current cardiac cycle to the first one
% in the next cycle. In this case, no RF pulses are performed during
% inversion or T2prep.

TNT_dur(1) = Delay_dur(1) + TFEPP_dur + TFEPP_delay_dur + T2prep_dur;
TNT_dur(2) = Delay_dur(2) + TFEPP_dur + TFEPP_delay_dur + T2prep_dur;
TNT_dur(3) = Delay_dur(3) + TFEPP_dur + TFEPP_delay_dur + T2prep_dur;
TNT_dur(4) = Delay_dur(4) + TFEPP_dur*0.5;
TNT_dur(5) = Delay_dur(5) + TFEPP_dur + T2prep_dur + TFEPP_delay_dur;
TNT_dur(6) = Delay_dur(6) + TFEPP_dur + T2prep_dur + TFEPP_delay_dur;
TNT_dur(7) = Delay_dur(7) + TFEPP_dur + TFEPP_delay_dur + T2prep_dur-30;
TNT_dur(8) = Delay_dur(8) + TFEPP_dur + TFEPP_delay_dur + T2prep_dur-50;
TNT_dur(9) = Delay_dur(9) + TFEPP_dur + TFEPP_delay_dur + T2prep_dur-70;
TNT_dur(10) = 0;

PP_delays = zeros(n_images,2);

PP_delay(1,1) = TFEPP_dur*0.5+TFEPP_delay_dur+T2prep_dur;
PP_delay(5,1) = TFEPP_dur*0.5+TFEPP_delay_dur+T2prep_dur;
PP_delay(8,2) = 30;
PP_delay(9,2) = 50;
PP_delay(10,2) = 70;

%% Calculate index of k0 acquisition of pulse train
k0_ind1 = length(SU_fa_list) + round(n_ex/2); % assume k0 at centre of readout train

k0_ind(1) = 1+k0_ind1;
k0_ind(2) = k0_ind1;
k0_ind(3) = k0_ind1;
k0_ind(4) = k0_ind1;
k0_ind(5) = 1+k0_ind1;
k0_ind(6) = k0_ind1;
k0_ind(7) = k0_ind1;
k0_ind(8) = 3+k0_ind1;
k0_ind(9) = 3+k0_ind1;
k0_ind(10) = 3+k0_ind1;

%% Define dictionaries for B1 calculation
% For speed, both T1 and T2 are coarsely sampled and centred around native myocardium
% T1 and T2.

T1_step = 100;
T2_step = 30;
B1_step = 0.05;

T1_start = 1000;
T2_start = 40;
B1_start = 0.5;

T1_end = 1500;
T2_end = 140;
B1_end = 1.0;

T1list=T1_start:T1_step:T1_end;
T2list=T2_start:T2_step:T2_end;
B1list=B1_start:B1_step:B1_end;
T1_l = length(T1list);
T2_l = length(T2list);
B1_l = length(B1list);

dictionary = zeros(T1_l*T2_l*B1_l, n_images);

Tlist=zeros(T1_l*T2_l*B1_l, 3);
i = 1;

for ind = 1:T1_l 
    for jnd = 1:T2_l
        for knd = 1:B1_l
            if T1list(ind) > T2list(jnd)
                Tlist(i,1) = T1list(ind); 
                Tlist(i,2) = T2list(jnd);
                Tlist(i,3) = B1list(knd);
                i = i + 1;
            end
        end
    end
end

%% determine optimal B1


parfor ind = 1:size(Tlist,1)
    sim_im_val = zeros(n_images,1);
    % scale the nominal flip angle (fa) with B1 factor in Tlist(ind,3)
    fa2 = fa*Tlist(ind,3);
    ss = fa2*ones([1 n_ex]);
    SU_fa_list2 = SU_fa_list*Tlist(ind,3);

    ss = [SU_fa_list2 ss];

    alphas = [];
    
    alphas(8,:) = d2r([t2prep_rf ss 0]);
    alphas(9,:) = d2r([t2prep_rf ss 0]);
    alphas(10,:) = d2r([t2prep_rf ss 0]);

    alphas(1,:) = d2r([inv_rf ss 0 NaN NaN]);
    alphas(2,:) = d2r([ss 0 NaN NaN NaN]);
    alphas(3,:) = d2r([ss 0 NaN NaN NaN]);
    alphas(4,:) = d2r([ss 0 NaN NaN NaN]);
    alphas(5,:) = d2r([inv_rf ss 0 NaN NaN]);
    alphas(6,:) = d2r([ss 0 NaN NaN NaN]);
    alphas(7,:) = d2r([ss 0 NaN NaN NaN]);
   
    Mz = 1;

    %Loop over the cardiac cyles. Only keep track of Mz across cardiac
    %cycles. Mxy is assumed to spoil between cycles.
    for im = 1:n_images
        n_rf = npulse;
        tmp = isnan(alphas(im,:));
        if sum(tmp) > 0
            tmp2 = find(tmp == 1);
            n_rf = min(n_rf, min(tmp2)-1);
        end
        
        %perform EPG
        [~,Fn1,Zn1] = EPG_GRE_dict(alphas(im,1:n_rf),phi(im,1:n_rf),TR,Tlist(ind,1),Tlist(ind,2),TNT_dur(im),PP_delay(im,:),Mz,'kmax',inf);
        
        ssfp = ifftshift(size(Fn1,1)*(ifft(ifftshift(Fn1,1),[],1)),1);
        phiTR= linspace(-pi,pi,size(ssfp,1));
        [~,idx0] = min(abs(phiTR));
        Mxy = abs(ssfp(idx0,:));
        sign_Mz = sign(Zn1(1,:));
        Mxy = (Mxy.*sign_Mz);
        Mz = Zn1(1,end);
        sim_im_val(im) = Mxy(k0_ind(im));
    end
    
    dictionary(ind,:) = sim_im_val;
        
end

dict_norm = zeros(T1_l*T2_l*B1_l, n_images);
sig_norm = zeros(n_images,1);

%% Dictionary matching, similar to MRF
for i=1:size(dictionary(:,:),1)
    dict_norm(i,:)=dictionary(i,:)./norm(dictionary(i,:));
end

for i=1:size(sig(:,:),2)
    sig_norm(:,i)=sig(:,i)./norm(sig(:,i));
end

innerproduct=dict_norm(:,:)*sig_norm(:,:);

[maxval,matchout]=max(abs(innerproduct));

B1map = Tlist(matchout(:),3);

B1map = reshape(B1map,[size(IM_phase_corrected,1) size(IM_phase_corrected,2)]);


%% draw ROI in septal myocardium to estimate B1 scaling factor
figure, imagesc(B1map);
title('Draw ROI in myocardial septum');

hFH = drawfreehand(); 
binaryImage_ROI = hFH.createMask();

close

measurements = regionprops(binaryImage_ROI, B1map, ...
    'MeanIntensity');
B1opt = measurements.MeanIntensity;

%% Apply estimated B1 and do the highly resolved T1 dictionary matching

fa2 = fa*B1opt;
ss = fa2*ones([1 n_ex]);
SU_fa_list2 = SU_fa_list*B1opt;

ss = [SU_fa_list2 ss];

alphas = [];

alphas(8,:) = d2r([t2prep_rf ss 0]);
alphas(9,:) = d2r([t2prep_rf ss 0]);
alphas(10,:) = d2r([t2prep_rf ss 0]);

alphas(1,:) = d2r([inv_rf ss 0 NaN NaN]);
alphas(2,:) = d2r([ss 0 NaN NaN NaN]);
alphas(3,:) = d2r([ss 0 NaN NaN NaN]);
alphas(4,:) = d2r([ss 0 NaN NaN NaN]);
alphas(5,:) = d2r([inv_rf ss 0 NaN NaN]);
alphas(6,:) = d2r([ss 0 NaN NaN NaN]);
alphas(7,:) = d2r([ss 0 NaN NaN NaN]);
   
T1_step = 1;
T2_step = 30;

T1_start = 200;
T2_start = 20;

T1list=T1_start:T1_step:2500;
T2list=T2_start:T2_step:150;
T1_l = length(T1list);
T2_l = length(T2list);

dictionary = zeros(T1_l*T2_l, n_images);

Tlist=zeros(T1_l*T2_l, 3);
i = 1;

for ind = 1:T1_l 
    for jnd = 1:T2_l
        if T1list(ind) > T2list(jnd)
            Tlist(i,1) = T1list(ind); 
            Tlist(i,2) = T2list(jnd);
            i = i + 1;
        end
    end
end

%% perform high resolution T1 dictionary + matching
tic
parfor ind = 1:size(Tlist,1)
    Mz = 1;
    sim_im_val = zeros(n_images,1);

    for im = 1:n_images
        n_rf = npulse;
        tmp = isnan(alphas(im,:));
        if sum(tmp) > 0
            tmp2 = find(tmp == 1);
            n_rf = min(n_rf, min(tmp2)-1);
        end
        
        [~,Fn1,Zn1] = EPG_GRE_dict(alphas(im,1:n_rf),phi(im,1:n_rf),TR,Tlist(ind,1),Tlist(ind,2),TNT_dur(im),PP_delay(im,:),Mz,'kmax',inf);

        ssfp = ifftshift(size(Fn1,1)*(ifft(ifftshift(Fn1,1),[],1)),1);
        phiTR= linspace(-pi,pi,size(ssfp,1));
        [~,idx0] = min(abs(phiTR));
        Mxy = abs(ssfp(idx0,:));
        sign_Mz = sign(Zn1(1,:));
        Mxy = (Mxy.*sign_Mz);
        Mz = Zn1(1,end);
        sim_im_val(im) = Mxy(k0_ind(im));

    end
    
    dictionary(ind,:) = sim_im_val;  
end


dict_norm = zeros(T1_l*T2_l, n_images);
sig_norm = zeros(n_images,1);

for i=1:size(dictionary(:,:),1)
    dict_norm(i,:)=dictionary(i,:)./norm(dictionary(i,:));
end

for i=1:size(sig(:,:),2)
    sig_norm(:,i)=sig(:,i)./norm(sig(:,i));
end

innerproduct=dict_norm(:,:)*sig_norm(:,:);

[maxval,matchout]=max(abs(innerproduct));

T1map = Tlist(matchout(:),1);
T2map = Tlist(matchout(:),2);

SI_M0 = max(sig(ref_im,:))*0.05;

for i = 1:size(sig,2)
    if sig(ref_im,i) < SI_M0
        T1map(i) = 0;
        T2map(i) = 0;
    end
end

T1map = reshape(T1map,[size(IM_phase_corrected,1) size(IM_phase_corrected,2)]);
T2map = reshape(T2map,[size(IM_phase_corrected,1) size(IM_phase_corrected,2)]);

T1map_HR = T1map;


%% Do the highly resolved T2 matching last

T1_step = 50;
T2_step = 1;

T1_start = 200;
T2_start = 1;

T1list=T1_start:T1_step:2500;
T2list=T2_start:T2_step:150;
T1_l = length(T1list);
T2_l = length(T2list);

dictionary = zeros(T1_l*T2_l, n_images);

Tlist=zeros(T1_l*T2_l, 3);
i = 1;

for ind = 1:T1_l 
    for jnd = 1:T2_l
        if T1list(ind) > T2list(jnd)
            Tlist(i,1) = T1list(ind); 
            Tlist(i,2) = T2list(jnd);
            i = i + 1;
        end
    end
end

%% perform high resolution T2 dictionary + matching

parfor ind = 1:size(Tlist,1)
    Mz = 1;
    sim_im_val = zeros(n_images,1);
    
    for im = 1:n_images
        n_rf = npulse;
        tmp = isnan(alphas(im,:));
        if sum(tmp) > 0
            tmp2 = find(tmp == 1);
            n_rf = min(n_rf, min(tmp2)-1);
        end
        
        [~,Fn1,Zn1] = EPG_GRE_dict(alphas(im,1:n_rf),phi(im,1:n_rf),TR,Tlist(ind,1),Tlist(ind,2),TNT_dur(im),PP_delay(im,:),Mz,'kmax',inf);
        ssfp = ifftshift(size(Fn1,1)*(ifft(ifftshift(Fn1,1),[],1)),1);
        phiTR= linspace(-pi,pi,size(ssfp,1));
        [~,idx0] = min(abs(phiTR));
        Mxy = abs(ssfp(idx0,:));
        sign_Mz = sign(Zn1(1,:));
        Mxy = (Mxy.*sign_Mz);
        Mz = Zn1(1,end);
        sim_im_val(im) = Mxy(k0_ind(im));

    end
    
    dictionary(ind,:) = sim_im_val;
        
end

dict_norm = zeros(T1_l*T2_l, n_images);
sig_norm = zeros(n_images,1);

for i=1:size(dictionary(:,:),1)
    dict_norm(i,:)=dictionary(i,:)./norm(dictionary(i,:));
end

for i=1:size(sig(:,:),2)
    sig_norm(:,i)=sig(:,i)./norm(sig(:,i));
end

innerproduct=dict_norm(:,:)*sig_norm(:,:);

[maxval,matchout]=max(abs(innerproduct));

T1map = Tlist(matchout(:),1);
T2map = Tlist(matchout(:),2);

SI_M0 = max(sig(ref_im,:))*0.05;

for i = 1:size(sig,2)
    if sig(ref_im,i) < SI_M0
        T1map(i) = 0;
        T2map(i) = 0;
    end
end
toc
T1map = reshape(T1map,[size(IM_phase_corrected,1) size(IM_phase_corrected,2)]);
T2map = reshape(T2map,[size(IM_phase_corrected,1) size(IM_phase_corrected,2)]);

T2map_HR = T2map; 
figure, 
subplot(1,2,1)
imagesc(T1map_HR)
title('T1 map')

subplot(1,2,2)
imagesc(T2map_HR)
title('T2 map')

