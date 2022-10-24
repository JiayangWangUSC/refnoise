%% effective NCC noise parameters estimation
datapath = '/project/jhaldar_118/jiayangw/dataset/brain_cppy/train/';
%datapath = '/home/wjy/Project/fastmri_dataset/miniset_brain_clean/';
dirname = dir(datapath);
N1 = 384; N2 = 396; Nc = 16; Ns = 8;

%%
for dir_num = 3:length(dirname)
    h5create([datapath,dirname(dir_num).name],'/ncc_effect',[N2,N1,2,Ns],'Datatype','single');
end

%%
fft2c = @(x) fftshift(fft2(ifftshift(x)))/sqrt(size(x,1)*size(x,2));
ifft2c = @(x) fftshift(ifft2(ifftshift(x)))*sqrt(size(x,1)*size(x,2)); 

%%
for dir_num=3:length(dirname)
    ncc_effect = zeros(N1,N2,2,Ns);
    kData = h5read([datapath,dirname(dir_num).name],'/kspace');
    kspace = complex(kData.r,kData.i)*2e5;
    kspace = permute(kspace,[4,2,1,3]);
    for s = 1:Ns
        kdata = reshape(kspace(s,:,:,:),2*N1,N2,Nc);
        im = ifft2c(kdata);
        im = im(192:575,:,:);

        patch = [reshape(im(1:50,1:50,:),[],Nc);reshape(im(end-50:end,1:50,:),[],Nc);reshape(im(1:50,end-50:end,:),[],Nc);reshape(im(end-50:end,end-50:end,:),[],Nc)].';
        cov = patch*patch'/size(patch,2);

        im =reshape(im,[],Nc);
        AT2 = sum(abs(im).^2,2);
        ASA = sum(abs(cov*im.').^2,1).';
        S = cov*cov;
        L = reshape((AT2*trace(S)+trace(S)^2)./(ASA+sum(abs(S(:)).^2)),N1,N2);
        V = trace(S)./L;
        ncc_effect(:,:,1,s) = L;
        ncc_effect(:,:,2,s) = V;
    end 
    ncc_effect = permute(ncc_effect,[2,1,3,4]);
    h5write([datapath,dirname(dir_num).name],'/ncc_effect',single(ncc_effect));
end
