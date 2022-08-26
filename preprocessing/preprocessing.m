%%
datapath = '/project/jhaldar_118/jiayangw/dataset/brain/train/';
dirname = dir(datapath);
N1 = 384; N2 = 396; Nc = 16; Ns = 8;

%%
newdatapath = '/project/jhaldar_118/jiayangw/dataset/brain_clean/train/';
for dir_num = 3:length(dirname)
    h5create([newdatapath,dirname(dir_num).name],'/kspace',[Ns,N1,N2,Nc]);
end

%%
fft2c = @(x) fftshift(fft2(ifftshift(x)))/sqrt(size(x(:),1))*4;
ifft2c = @(x) fftshift(ifft2(ifftshift(x)))*sqrt(size(x(:),1))/4; 

%%
for dir_num = 3:length(dirname)
%% slice selection, undersampling and whitening 
kspace = h5read([datapath,dirname(dir_num).name],'/kspace');
kspace = complex(kspace.r,kspace.i);
kspace = permute(kspace,[4,2,1,3]);

kdata = reshape(kspace(1,2*(1:N1),:,:),N1,N2,Nc);
im = ifft2c(kdata);
patch = [reshape(im(1:50,1:50,:),[],Nc);reshape(im(end-50:end,1:50,:),[],Nc);reshape(im(1:50,end-50:end,:),[],Nc);reshape(im(end-50:end,end-50:end,:),[],Nc)].';
cov = patch*patch'/size(patch,2);
cov_inv = inv(cov);
[U,S,V] = svd(cov_inv);
W = (U*sqrt(S))';

kspace = kspace(1:Ns,2*(1:N1),:,:);
kspace = W * reshape(kspace,[],Nc).';
kspace = reshape(kspace.',Ns,N1,N2,Nc);

%% denoising
for s = 1:Ns
    kdata = reshape(kspace(s,:,:,:),N1,N2,Nc);
    im = ifft2c(kdata);
    input = zeros(N1,N2,2*Nc);
    input(:,:,1:Nc) = real(im);
    input(:,:,Nc+1:2*Nc) = imag(im);
    patch = reshape(input(1:50,1:50,:),[],2*Nc).';
    cov = 0.5*eye(2*Nc);
    output = OWT_MC_SURELET_denoise(input,'sym8',cov);
    output = complex(output(:,:,1:Nc),output(:,:,Nc+1:2*Nc));
    kdata = fft2c(output);
    kspace(s,:,:,:) = kdata;
end

%% new dataset
h5write([newdatapath,dirname(dir_num).name],'/kspace',kspace);
end
