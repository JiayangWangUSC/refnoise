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

def toIm(kspace): 
    image = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace)), dim=1)
    return image

# %% sampling mask
mask = torch.zeros(ny)
mask[torch.arange(132)*3] = 1
mask[torch.arange(186,210)] =1
mask = mask.bool().unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(nc,nx,1,2)

# %% unet loader
unet = Unet(
  in_chans = 32,
  out_chans = 32,
  chans = 256,
  num_pool_layers = 4,
  drop_prob = 0.0
)
unet = torch.load('/home/wjy/Project/refnoise_model/unet_noisy',map_location=torch.device('cpu'))

# %%
with torch.no_grad():
    kspace = test_data[0].unsqueeze(0)
    noise = math.sqrt(0.5)*torch.randn_like(kspace)
    kspace_noise = kspace + noise

    gt = toIm(kspace)
    gt_noise = toIm(kspace_noise)

    Mask = mask.unsqueeze(0)
    kspace_undersample = torch.mul(kspace_noise,Mask)

    image = fastmri.ifft2c(torch.mul(Mask,kspace_undersample))
    image_input = torch.cat((image[:,:,:,:,0],image[:,:,:,:,1]),1) 
    image_output = unet(image_input)
    image_recon = torch.cat((image_output[:,torch.arange(nc),:,:].unsqueeze(4),image_output[:,torch.arange(nc,2*nc),:,:].unsqueeze(4)),4)
    recon = fastmri.rss(fastmri.complex_abs(image_recon), dim=1)

    plt.imshow(recon.detach().squeeze()/50,cmap='gray',vmax=1)
# %%
