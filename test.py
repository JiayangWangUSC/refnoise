# %%
import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.data import transforms
from fastmri.models import Unet
from varnet import *
import pathlib
import numpy as np
import torch.optim as optim
from fastmri.data import  mri_data
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import math

# %% data loader
from my_data import *

nc = 16
nx = 384
ny = 396

def data_transform(kspace, mask, target, data_attributes, filename, slice_num):
    # Transform the kspace to tensor format
    kspace = transforms.to_tensor(kspace)
    kspace = torch.cat((kspace[torch.arange(nc),:,:].unsqueeze(-1),kspace[torch.arange(nc,2*nc),:,:].unsqueeze(-1)),-1)
    return kspace

test_data = SliceDataset(
    root=pathlib.Path('/home/wjy/Project/fastmri_dataset/miniset_brain_clean/'),
    #root = pathlib.Path('/project/jhaldar_118/jiayangw/dataset/brain_clean/train/'),
    transform=data_transform,
    challenge='multicoil'
)

def KtoIm(kspace): 
    image = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace)), dim=1)
    return image

def MtoIm(im):
    Im = fastmri.rss(fastmri.complex_abs(im),dim=1)
    return Im

# %% sampling mask
mask = torch.zeros(ny)
mask[torch.arange(132)*3] = 1
mask[torch.arange(186,210)] =1
mask = mask.bool().unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(nc,nx,1,2)


# %% imnet loader
imnet = torch.load('/home/wjy/Project/refnoise_model/imnet_noisy',map_location=torch.device('cpu'))

# %%
with torch.no_grad():
    kspace = test_data[1].unsqueeze(0)
    noise = math.sqrt(0.5)*torch.randn_like(kspace)
    kspace_noise = kspace + noise

    gt = fastmri.ifft2c(kspace)
    gt_noise = fastmri.ifft2c(kspace_noise)

    Mask = mask.unsqueeze(0)
    kspace_undersample = torch.mul(kspace_noise,Mask)

    image = fastmri.ifft2c(kspace_undersample)
    image_input = torch.cat((image[:,:,:,:,0],image[:,:,:,:,1]),1) 
    image_output = imnet(image_input)
    recon = torch.cat((image_output[:,torch.arange(nc),:,:].unsqueeze(4),image_output[:,torch.arange(nc,2*nc),:,:].unsqueeze(4)),4)

# %%
#plt.imshow(recon.detach().squeeze()/50,cmap='gray',vmax=1)
L2Loss = torch.nn.MSELoss()
mse = L2Loss(recon,gt)
mse_approx = L2Loss(recon,gt_noise)

print(mse)
print(mse_approx)

# %% SURE
epsilon = 1e-3
mc_noise = math.sqrt(0.5)*torch.randn_like(kspace_noise)
kspace_mc = torch.mul(kspace_noise + mc_noise, Mask)
image = fastmri.ifft2c(kspace_mc)
image_input = torch.cat((image[:,:,:,:,0],image[:,:,:,:,1]),1) 
image_output = imnet(image_input)
recon_mc = torch.cat((image_output[:,torch.arange(nc),:,:].unsqueeze(4),image_output[:,torch.arange(nc,2*nc),:,:].unsqueeze(4)),4)


# %%
MSE = torch.zeros(3,10)
MSE_approx = torch.zeros(3,10)
SURE = torch.zeros(3,10)


# %% varnet loader
epoch = 190
sigma = 1
cascades = 6
chans = 20
#varnet = torch.load("/home/wjy/Project/refnoise_model/varnet_l2mc_noise"+str(sigma)+"_cascades"+str(cascades)+"_channels"+str(chans)+"_epoch"+str(epoch),map_location = 'cpu')
varnet = torch.load("/home/wjy/Project/refnoise_model/varnet_noisy_cascades"+str(cascades)+"_channels"+str(chans)+"_epoch"+str(epoch),map_location = 'cpu')

# %%
with torch.no_grad():
    kspace = test_data[0].unsqueeze(0)
    noise = sigma*math.sqrt(0.5)*torch.randn_like(kspace)
    kspace_noise = kspace + noise

    gt = fastmri.ifft2c(kspace)
    gt_noise = fastmri.ifft2c(kspace_noise)

    Mask = mask.unsqueeze(0)
    kspace_undersample = torch.mul(kspace_noise,Mask)
    
    recon = varnet(kspace_undersample, Mask, 24)

# %%
#plt.imshow(recon.detach().squeeze()/50,cmap='gray',vmax=1)
L2Loss = torch.nn.MSELoss()
mse = torch.sum(torch.square(recon-gt))/nx/ny/nc
mse_approx = torch.sum(torch.square(recon-gt_noise))/nx/ny/nc

print(mse)
print(mse_approx)
# %%
nrmse = torch.norm(MtoIm(recon)-MtoIm(gt))/torch.norm(MtoIm(gt))
nrmse_approx = torch.norm(MtoIm(recon)-MtoIm(gt_noise))/torch.norm(MtoIm(gt_noise))
print(nrmse)
print(nrmse_approx)

# %% SURE
epsilon = 1e-6
mc_kspace = math.sqrt(0.5)*torch.randn_like(kspace_noise)
mc_image = fastmri.ifft2c(mc_kspace)
kspace_mc = torch.mul(kspace_noise+epsilon*mc_kspace, Mask)
recon_mc = varnet(kspace_mc, Mask, 24)

div = torch.sum(torch.mul(mc_image, recon_mc-recon))/epsilon
sure = mse_approx - sigma*sigma + 2*sigma*sigma/(nx*ny*nc)*div
print(sure)
# %%
