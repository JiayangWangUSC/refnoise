# %%
import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.data import transforms
from fastmri.models import Unet
import pathlib
import numpy as np
import torch.optim as optim
from fastmri.data import  mri_data
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# %% data loader
def data_transform(kspace, mask, target, data_attributes, filename, slice_num):
    # Transform the kspace to tensor format
    kspace = transforms.to_tensor(kspace)
    image = fastmri.ifft2c(kspace)
    image = image[:,torch.arange(191,575),:,:]
    kspace = fastmri.fft2c(image)/5e-5
    return kspace

test_data = mri_data.SliceDataset(
    root=pathlib.Path('/home/wjy/Project/fastmri_dataset/test_brain/'),
    #root = pathlib.Path('/project/jhaldar_118/jiayangw/dataset/brain/train/'),
    transform=data_transform,
    challenge='multicoil'
)


def toIm(kspace): 
    image = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace)), dim=1)
    return image

# %%
model_pre = torch.load('/home/wjy/Project/optsamp_models/uni_model_sigma1',map_location=torch.device('cpu'))
model_clean = torch.load('/home/wjy/Project/refnoise_model/model_l2_clean',map_location=torch.device('cpu'))
model_noise1 = torch.load('/home/wjy/Project/refnoise_model/model_l2_noise0.2',map_location=torch.device('cpu'))
model_noise2 = torch.load('/home/wjy/Project/refnoise_model/model_l2_noise0.4',map_location=torch.device('cpu'))

# %%
mask = torch.zeros(396)
mask[torch.arange(130)*3] = 1
mask[torch.arange(186,210)] =1
Mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(4).repeat(1,16,384,1,2)

# %%
batch_size = 16
test_dataloader = torch.utils.data.DataLoader(test_data,batch_size,shuffle=True)


# %%
loss1 = torch.zeros(100*batch_size,2)
loss2 = torch.zeros(100*batch_size,2)
# %%
batch_count = 0
with torch.no_grad():
    for test_batch in test_dataloader:
        print(batch_count)
        
        image = fastmri.ifft2c(test_batch)
        image_input = torch.cat((image[:,:,:,:,0],image[:,:,:,:,1]),1) 
        image_output = model_pre(image_input)
        image_pre = torch.cat((image_output[:,torch.arange(16),:,:].unsqueeze(4),image_output[:,torch.arange(16,32),:,:].unsqueeze(4)),4)
        kspace = fastmri.fft2c(image_pre)
        im_clean = toIm(kspace)

        torch.manual_seed(batch_count)
        noise = torch.rand_like(kspace)

        im_noise = toIm(kspace+0.1*noise)
    
       # image = fastmri.ifft2c(torch.mul(Mask,kspace))
       # image_input = torch.cat((image[:,:,:,:,0],image[:,:,:,:,1]),1) 
       # image_output = model_clean(image_input)
       # image_recon = torch.cat((image_output[:,torch.arange(16),:,:].unsqueeze(4),image_output[:,torch.arange(16,32),:,:].unsqueeze(4)),4)
       # recon0 = fastmri.rss(fastmri.complex_abs(image_recon), dim=1)

        image = fastmri.ifft2c(torch.mul(Mask,kspace+0.1*noise))
        image_input = torch.cat((image[:,:,:,:,0],image[:,:,:,:,1]),1) 
        image_output = model_noise1(image_input)
        image_recon = torch.cat((image_output[:,torch.arange(16),:,:].unsqueeze(4),image_output[:,torch.arange(16,32),:,:].unsqueeze(4)),4)
        recon1 = fastmri.rss(fastmri.complex_abs(image_recon), dim=1)

      #  loss1[batch_count,0] = torch.norm(recon0-im_clean)
      #  loss1[batch_count,1] = torch.norm(recon0-im_noise)
        for image_num in range(batch_size):
            loss2[batch_count*batch_size+image_num,0] = torch.norm(recon1[image_num,:,:]-im_clean[image_num,:,:])/torch.norm(im_clean[image_num,:,:])
            loss2[batch_count*batch_size+image_num,1] = torch.norm(recon1[image_num,:,:]-im_noise[image_num,:,:])/torch.norm(im_noise[image_num,:,:])

        batch_count = batch_count+1


# %%
kspace = test_data[0].unsqueeze(0)
im_orig = toIm(kspace)
image = fastmri.ifft2c(kspace)
image_input = torch.cat((image[:,:,:,:,0],image[:,:,:,:,1]),1) 
image_output = model_pre(image_input)
image_pre = torch.cat((image_output[:,torch.arange(16),:,:].unsqueeze(4),image_output[:,torch.arange(16,32),:,:].unsqueeze(4)),4)
kspace = fastmri.fft2c(image_pre)
im_clean = toIm(kspace)
plt.imshow(im_clean.detach().squeeze(),cmap='gray')
# %%
noise = torch.rand_like(kspace)
im_noise1 = toIm(kspace+0.1*noise)
im_noise2 = toIm(kspace+0.4*noise)
# %%

image = fastmri.ifft2c(torch.mul(Mask,kspace))
image_input = torch.cat((image[:,:,:,:,0],image[:,:,:,:,1]),1) 
image_output = model_clean(image_input)
image_recon = torch.cat((image_output[:,torch.arange(16),:,:].unsqueeze(4),image_output[:,torch.arange(16,32),:,:].unsqueeze(4)),4)
recon0 = fastmri.rss(fastmri.complex_abs(image_recon), dim=1)

# %%
image = fastmri.ifft2c(torch.mul(Mask,kspace))
image_input = torch.cat((image[:,:,:,:,0],image[:,:,:,:,1]),1) 
image_output = model_noise1(image_input)
image_recon = torch.cat((image_output[:,torch.arange(16),:,:].unsqueeze(4),image_output[:,torch.arange(16,32),:,:].unsqueeze(4)),4)
recon1 = fastmri.rss(fastmri.complex_abs(image_recon), dim=1)


