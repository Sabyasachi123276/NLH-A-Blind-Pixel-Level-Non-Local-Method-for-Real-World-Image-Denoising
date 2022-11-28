I=imread("house.png");
%I=imread("Cameraman256.png");% original image
I=im2double(I);
[M,N]=size(I);
randn('seed', 0);%randn(generator,sd)

for sigma=[5,15,25,35,50,75];
im=I+(sigma/255)*randn(size(I)); % adding noise with normally distributed random number 
PSNR = psnr(I,double(im));  
SSIM = ssim(I,double(im));

%% Noise level estimation 
[~, sigma_est, ~] = noise_estimation(im);
sigma_est = double(sigma_est);
% 8*8  patch window size 
% search its 16 most similar patches (including...
%the reference patch itself in the image by selecting the 16 minimum Euclidean distances which are computed between
%the reference patch and each patch in a 39×39 neighborhood.
[dis_map,~] = NL_distance(8,16,2,39,single(8),double(im));
dis_map = double(dis_map);
%imshow(dis_map)

%%In all experiments, we set W = 40, m = 16, q = 4, τ = 2, λ = 0.6. For synthetic
%AWGN corrupted image denoising, we set √n = 8, K = 4 for 0 < σ ≤ 50, √n = 10, K = 5 for σ > 50 in both stages.

%%%%%%%%%%%%%%%%%%
Ns     = 43;% EACH PATCH NEIGHBORHOOD
N3     = 4;% patch window size
N2     = 16;% number of pixel in a patch
%%%%%%%%%%
%Main function -add the denoised image yk−1 back to the original noisy image 'I' and obtain the noisy image 'im' as
imr = im;  

alpha = 0.618; 

lamda = 0.8; %regularization  parameter

Thr = 1.45;  %Hard threshold parameter

% k times - iteration number  and thresholding
if sigma_est < 7.5
    k = 2;
    N_step = 5;
    N1     = 9;
elseif sigma_est >= 7.5 && sigma_est < 12.5
    k = 3;
    N_step = 5;
    N1     = 9;
 elseif sigma_est >= 12.5 && sigma_est < 35
    k = 3;
    N_step = 6;
    N1     = 9;
 elseif sigma_est >= 35 && sigma_est < 55
    k = 4;
    N_step = 6;
    N1     = 9;
  elseif sigma_est >= 55 && sigma_est < 75
      k = 5;
     N_step = 7;
     N1     = 10;
 elseif sigma_est >= 75 && sigma_est < 85
     k = 6;
      N_step = 8;
      N1     = 11;
else 
      k = 7;
      N_step = 9;
      N1     = 11;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% im = λ*denoised image + (1 − λ) original imahe.

for j = 1:k 
    imr = NLH_AWGN_Gray(N1,N2,N3,Ns,N_step,double(alpha*imr+(1-alpha)*im),single(Thr),...
        sigma_est, double(lamda*imr+(1-lamda)*im));
    PSNR = 10*log10(1/mean((I(:)-double(im(:))).^2));
 
    %figure,imshow(double(imr));

    %fprintf( 'Iter = %2.0f, PSNR = %2.4f \n\n\n', j, PSNR );

     N_step = N_step -1;
     if N_step <= 3 || j==k
         N_step = 3;
     end
 
end 

imr = double(imr);
PSNR = psnr(I,double(imr));  
SSIM = ssim(I,double(imr));  
%figure,imshow(imr);
%fprintf('Basic ESTIMATE, PSNR = %2.4f, SSIM = %2.4f \n', PSNR, SSIM);

%Noise level estimation
[~, sigma_est1, ~] = noise_estimation(imr); %% estimate noisy level
sigma_est1 = double(sigma_est1);

% NLH_wiener_filter
N2 = 64; %NUMBER OF PIXEL IN 8*8 PATCH WINDOW
N3 = 8;% PATCH WINDOW SIZE
Ns = 129;

if sigma_est < 25
    N1      = 8;
    N_step1 = 8;
    N_step2 = 5;
elseif sigma_est >= 25 && sigma_est < 75
    N1      = 16;
    N_step1 = 16;
    N_step2 = 13;    
else
    N1      = 24;
    N_step1 = 24;
    N_step2 = 23;
end


beta = 0.8;
gamma=2.3;
%%% Stage two: Wiener filtering
y_est = NLH_AWGN_Wiener_Gray(N1,N2,N3,Ns,N_step1,double(im),single(gamma),...
                sigma_est/255, double(dis_map),beta,double(imr*gamma),double(imr*beta+im*(1-beta)));            
    

                
y_est = NLH_AWGN_Wiener_Gray(N1,N2,N3,Ns,N_step2,double(y_est),single(gamma),sigma_est/255, ...
    double(dis_map),beta,double(imr*gamma),double(y_est*beta+im*(1-beta)));
PSNR = psnr(I,double(y_est));        
%figure,imshow(y_est);

% Calculate MMSIM value
SSIM = ssim(I,double(y_est));

fprintf('FINAL ESTIMATE, PSNR = %2.4f, SSIM = %2.4f \n', PSNR, SSIM);

figure,
subplot(131)
imshow(I)
title("Original Image")
subplot(132)
imshow(im)
title("Noisy Image")
subplot(133)
imshow(y_est)
title("Denoised Image")
MSE=immse(I,im2double(y_est))

end



