%%
datapath = '/project/jhaldar_118/jiayangw/dataset/brain_clean/train/';
%datapath = '/home/wjy/Project/fastmri_dataset/miniset_brain_clean/';
dirname = dir(datapath);
N1 = 384; N2 = 396; Nc = 16; Ns = 8;

%%
%newdatapath = '/project/jhaldar_118/jiayangw/dataset/brain_clean/train/';
%for dir_num = 3:length(dirname)
%    h5create([datapath,dirname(dir_num).name],'/kspace_clean',[N2,N1,2*Nc,Ns],'Datatype','single');
%end

%%
fft2c = @(x) fftshift(fft2(ifftshift(x)))/sqrt(size(x,1)*size(x,2));
ifft2c = @(x) fftshift(ifft2(ifftshift(x)))*sqrt(size(x,1)*size(x,2)); 

%%
for dir_num = 3:length(dirname)
%% slice selection, undersampling and whitening 
kData = h5read([datapath,dirname(dir_num).name],'/kspace');
kspace = complex(kData.r,kData.i);
kspace = permute(kspace,[4,2,1,3]);

kdata = reshape(kspace(1,:,:,:),2*N1,N2,Nc);
im = ifft2c(kdata);
im = im(192:575,:,:);

patch = [reshape(im(1:50,1:50,:),[],Nc);reshape(im(end-50:end,1:50,:),[],Nc);reshape(im(1:50,end-50:end,:),[],Nc);reshape(im(end-50:end,end-50:end,:),[],Nc)].';
cov = patch*patch'/size(patch,2);
cov_inv = inv(cov);
[U,S,V] = svd(cov_inv);
W = (U*sqrt(S))';
WW = U * sqrt(inv(S));

kspace_new = zeros(Ns,N1,N2,Nc);
%% denoising
for s = 1:Ns
    kdata = reshape(kspace(s,:,:,:),2*N1,N2,Nc);
    im = ifft2c(kdata);
    im = im(192:575,:,:);
    im = reshape((W*reshape(im,[],Nc).').',N1,N2,Nc);
    
    input = zeros(N1,N2,2*Nc);
    input(:,:,1:Nc) = real(im);
    input(:,:,Nc+1:2*Nc) = imag(im);
    cov = 0.5*eye(2*Nc);
    output = OWT_MC_SURELET_denoise(input,'sym8',cov);
    output = complex(output(:,:,1:Nc),output(:,:,Nc+1:2*Nc));
    output = output + sqrt(0.5)*complex(randn(size(output)),randn(size(output)));
    output = reshape((WW*reshape(output,[],Nc).').',N1,N2,Nc);
    
    kdata = fft2c(output);
    kspace_new(s,:,:,:) = kdata;
end
kspace_new = permute(kspace_new,[3,2,4,1]);
%% new dataset
kdata = zeros(N2,N1,2*Nc,Ns);
kdata(:,:,1:Nc,:) = real(kspace_new);
kdata(:,:,Nc+1:2*Nc,:) = imag(kspace_new);
kdata = single(kdata);
h5write([datapath,dirname(dir_num).name],'/kspace_clean',kdata);
end
