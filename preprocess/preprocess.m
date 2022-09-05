%%
datapath = '/project/jhaldar_118/jiayangw/dataset/brain/train/';
dirname = dir(datapath);
N1 = 384; N2 = 396; Nc = 16; Ns = 8; Nt = 8;

%%
newdatapath = '/project/jhaldar_118/jiayangw/dataset/brain_clean/train/';
%for dir_num = 3:length(dirname)
%    h5create([newdatapath,dirname(dir_num).name],'/kspace',[Ns,N1,N2,Nc,2]);
%end

%%
fft2c = @(x) fftshift(fft2(ifftshift(x)))/sqrt(size(x,1)*size(x,2))*4;
ifft2c = @(x) fftshift(ifft2(ifftshift(x)))*sqrt(size(x,1)*size(x,2))/4; 

%%
for dir_num = 3:length(dirname)
%% slice selection, undersampling and whitening 
kspace = h5read([datapath,dirname(dir_num).name],'/kspace');
kspace = complex(kspace.r,kspace.i);
kspace = permute(kspace,[4,2,1,3]);

kdata = reshape(kspace(1,:,:,:),2*N1,N2,Nc);
im = ifft2c(kdata);
im = im(192:575,:,:);
im = reshape(im,[],Nc);
[~,~,T] = svd(im'*im);
T = T(:,1:Nt);
im = im*T;
im = reshape(im,N1,N2,Nt);

patch = [reshape(im(1:50,1:50,:),[],Nt);reshape(im(end-50:end,1:50,:),[],Nt);reshape(im(1:50,end-50:end,:),[],Nt);reshape(im(end-50:end,end-50:end,:),[],Nt)].';
cov = patch*patch'/size(patch,2);
cov_inv = inv(cov);
[U,S,V] = svd(cov_inv);
W = (U*sqrt(S))';


kspace_new = zeros(Ns,N1,N2,Nc);
%% denoising
for s = 1:Ns
    kdata = reshape(kspace(s,:,:,:),2*N1,N2,Nc);
    im = ifft2c(kdata);
    im = im(192:575,:,:);
    im = reshape((W*(reshape(im,[],Nc)*T).').',N1,N2,Nt);
    
    input = zeros(N1,N2,2*Nt);
    input(:,:,1:Nt) = real(im);
    input(:,:,Nt+1:2*Nt) = imag(im);
    cov = 0.5*eye(2*Nt);
    output = OWT_MC_SURELET_denoise(input,'sym8',cov);
    output = complex(output(:,:,1:Nt),output(:,:,Nt+1:2*Nt));
    output = reshape(reshape(output,[],Nt)*T',N1,N2,Nc);
    kdata = fft2c(output);
    kspace_new(s,:,:,:) = kdata;
end

%% new dataset
kdata = zeros(Ns,N1,N2,Nc,2);
kdata(:,:,:,:,1) = real(kspace_new);
kdata(:,:,:,:,2) = imag(kspace_new);
h5write([newdatapath,dirname(dir_num).name],'/kspace',kdata);
end
