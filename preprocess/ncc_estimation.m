%% effective NCC noise parameters estimation
datapath = '/project/jhaldar_118/jiayangw/dataset/brain_copy/train/';
%datapath = '/home/wjy/Project/fastmri_dataset/brain_copy/';
dirname = dir(datapath);
N1 = 384; N2 = 396; Nc = 16; Ns = 8;

%%
%for dir_num = 3:length(dirname)
%    h5create([datapath,dirname(dir_num).name],'/ncc_effect',[N2,N1,2,Ns],'Datatype','single');
%end

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
        %cov = (real(patch)*real(patch)'/size(patch,2)+imag(patch)*imag(patch)'/size(patch,2))/2;
        cov = patch*patch'/size(patch,2);
%%  
         im =reshape(im,[],Nc);
         ML2 = reshape(sum(abs(im).^2,2),N1,N2);
         SAS = reshape(sum(conj(im).*(cov*im.').',2),N1,N2);
         
%%
         ML2_avg = ML2;
         SAS_avg = SAS;
         R = 1;
         for x = 1+R:N1-R
            for y = 1+R:N2-R
               ML2_avg(x,y) = sum(sum(ML2(x-R:x+R,y-R:y+R),1),2)/(2*R+1)^2;
               SAS_avg(x,y) = sum(sum(SAS(x-R:x+R,y-R:y+R),1),2)/(2*R+1)^2;
            end
         end
         
         %%
         ML2_avg = reshape(ML2_avg,[],1);
         SAS_avg = reshape(SAS_avg,[],1);
         support = sqrt(ML2_avg);
         support(support < 2*sqrt(trace(cov)))=0;
         support(support > 0)=1;
         mask = ones(N1,N2);
         mask(1:R,:) = 0; mask(:,1:R) = 0; mask(N1-R+1:N1,:) = 0; mask(:,N2-R+1:N2) = 0;   
         mask = mask(:);
        
        L = (2*ML2_avg*trace(cov)-trace(cov)^2)./(2*SAS_avg-sum(abs(cov(:)).^2));
        L = support.*L + (1-support) * (trace(cov)^2/sum(abs(cov(:)).^2));
        L = real(reshape(L,N1,N2));
        V = real(trace(cov)./L);
        ncc_effect(:,:,1,s) = L;
        ncc_effect(:,:,2,s) = V;
    end 
    ncc_effect = permute(ncc_effect,[2,1,3,4]);
    h5write([datapath,dirname(dir_num).name],'/ncc_effect',single(ncc_effect));
end
