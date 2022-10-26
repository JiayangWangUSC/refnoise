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
from my_data import SliceDataset

nc = 16
nx = 384
ny = 396

def data_transform(kspace_noisy, kspace_clean, ncc_effect):
    # Transform the kspace to tensor format
    ncc_effect = transforms.to_tensor(ncc_effect)
    kspace_noisy = transforms.to_tensor(kspace_noisy)
    kspace_noisy = torch.cat((kspace_noisy[torch.arange(nc),:,:].unsqueeze(-1),kspace_noisy[torch.arange(nc,2*nc),:,:].unsqueeze(-1)),-1)
    kspace_clean = transforms.to_tensor(kspace_clean)
    kspace_clean = torch.cat((kspace_clean[torch.arange(nc),:,:].unsqueeze(-1),kspace_clean[torch.arange(nc,2*nc),:,:].unsqueeze(-1)),-1)

    return kspace_noisy, kspace_clean, ncc_effect

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
mask[torch.arange(99)*4] = 1
mask[torch.arange(186,210)] =1
mask = mask.bool().unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(nc,nx,1,2)


# %% imnet loader
imnet = torch.load('/home/wjy/Project/refnoise_model/imnet_mse',map_location=torch.device('cpu'))

# %%
with torch.no_grad():
    kspace = test_data[0].unsqueeze(0)
    noise = math.sqrt(0.5)*torch.randn_like(kspace)
    kspace_noise = kspace + noise

    gt = KtoIm(kspace)
    gt_noise = KtoIm(kspace_noise)

    Mask = mask.unsqueeze(0)
    kspace_undersample = torch.mul(kspace_noise,Mask)

    image = fastmri.ifft2c(kspace_undersample)
    image_input = torch.cat((image[:,:,:,:,0],image[:,:,:,:,1]),1) 
    image_output = imnet(image_input)
    recon = torch.cat((image_output[:,torch.arange(nc),:,:].unsqueeze(4),image_output[:,torch.arange(nc,2*nc),:,:].unsqueeze(4)),4)

    recon = MtoIm(recon)

# %% varnet loader
epoch = 100
sigma = 1
cascades = 8
chans = 16
varnet = torch.load("/home/wjy/Project/refnoise_model/varnet_mae_acc4_cascades"+str(cascades)+"_channels"+str(chans)+"_epoch"+str(epoch),map_location = 'cpu')

# %%
with torch.no_grad():
    kspace_noisy, kspace_clean, ncc_effect = test_data[0]
    noise = sigma*math.sqrt(0.5)*torch.randn_like(kspace)
    kspace_noise = kspace + noise

    gt = KtoIm(kspace)
    gt_noise = KtoIm(kspace_noise)

    Mask = mask.unsqueeze(0)
    kspace_undersample = torch.mul(kspace_noise,Mask)
    
    recon_M = varnet(kspace_undersample, Mask, 24)

    recon = MtoIm(recon_M)
# %%
L1Loss = torch.nn.L1Loss()
L2Loss = torch.nn.MSELoss()


# %%
import scipy.special as ss

def NccLoss(x1,x2,sigma,nc):
    x = x1*x2/(sigma*sigma/2)
    y = torch.sum(torch.square(x1)/(sigma*sigma)-(torch.log(ss.ive(nc-1,x))+x)+(nc-1)*torch.log(x1))
    return y/torch.sum(torch.ones_like(x))


# %%
sp = torch.ge(gt,0.03*torch.max(gt))
print(L2Loss(recon,gt))
print(L2Loss(torch.mul(recon,sp),torch.mul(gt,sp)))
print(L2Loss(recon,gt_noise))
print(L1Loss(recon,gt))
print(L1Loss(torch.mul(recon,sp),torch.mul(gt,sp)))
print(L1Loss(recon,gt_noise))
print(NccLoss(recon,gt,sigma,nc)-NccLoss(gt, gt,sigma,nc))
print(NccLoss(recon,gt_noise,sigma,nc)-NccLoss(gt_noise,gt_noise,sigma,nc))

# %%
