%% effective NCC noise parameters estimation
datapath = '/home/wjy/Project/fastmri_dataset/test_brain/';
dirname = dir(datapath);
N1 = 384; N2 = 396; Nc = 16; Ns = 8;

fft2c = @(x) fftshift(fft2(ifftshift(x)))/sqrt(size(x,1)*size(x,2));
ifft2c = @(x) fftshift(ifft2(ifftshift(x)))*sqrt(size(x,1)*size(x,2)); 

%%
sigma = zeros(1,45);
L_est = zeros(1,45);
for dir_num=3:length(dirname)

slice_num = 1;

kData = h5read([datapath,dirname(dir_num).name],'/kspace');
kspace = complex(kData.r,kData.i)*2e5;
kspace = permute(kspace,[4,2,1,3]);

kdata = reshape(kspace(slice_num,:,:,:),2*N1,N2,Nc);
im = ifft2c(kdata);
im = im(192:575,:,:);

patch = [reshape(im(1:50,1:50,:),[],Nc);reshape(im(end-50:end,1:50,:),[],Nc);reshape(im(1:50,end-50:end,:),[],Nc);reshape(im(end-50:end,end-50:end,:),[],Nc)].';
cov = patch*patch'/size(patch,2);


im =reshape(im,[],Nc);
AT2 = sum(abs(im).^2,2);
ASA = sum(abs(cov*im.').^2,1).';
S = cov*cov;
L = (AT2*trace(S)+trace(S)^2)./(ASA+sum(abs(S(:)).^2));
support = sqrt(sum(abs(im).^2,3));
support = (support>0.05*max(support(:)));
L_est(dir_num-2) = sum(L.*support)/sum(support);

sigma(dir_num-2)=mean(diag(cov));

end

%%
